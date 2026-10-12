"""Train and evaluate corrected ISTA-Net+ cardiac MRI experiments.

The script supports two modes:

* ``fixed``: a deterministic variable-density Cartesian mask;
* ``learnable``: a hard-budget STE mask jointly optimized with ISTA-Net+.

Fixed measurements are cached. For joint training, measurements are generated
inside the autograd graph. The Cartesian path stores the complex adjoint image
instead of 2-D k-space; the two representations define the same normal operator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from ista_net import (  # noqa: E402
    ISTANetPlus,
    LearnableMask,
    count_params,
    simulate_cartesian_measurement,
)
from sampling import (  # noqa: E402
    body_mask,
    cartesian_mask,
    psnr,
    psnr_masked,
    ssim,
    ssim_masked,
)


class ReconSet(Dataset):
    """Load normalized slices and optionally cache fixed-mask measurements."""

    def __init__(
        self,
        slice_dir: Path,
        case_ids: Iterable[str],
        mask: Optional[np.ndarray] = None,
        fft_batch: int = 64,
    ):
        self.imgs = []
        self.case_of = []
        for cid in case_ids:
            d = np.load(slice_dir / f"{cid}.npz")
            imgs = d["img"].astype(np.float32, copy=False)
            self.imgs.append(imgs)
            self.case_of.extend([cid] * len(imgs))
        self.imgs = np.concatenate(self.imgs, axis=0)
        if len(self.case_of) != len(self.imgs):
            raise RuntimeError("case id count does not match slice count")
        self.zf = None
        self.k_meas = None
        if mask is not None:
            self._cache_measurements(mask, fft_batch)

    def _cache_measurements(self, mask: np.ndarray, fft_batch: int) -> None:
        """Precompute fixed-mask inputs once per ratio."""
        n, h, w = self.imgs.shape
        self.zf = np.empty((n, h, w), dtype=np.float32)
        self.k_meas = np.empty((n, 2, h, w), dtype=np.float32)
        mask = mask.astype(np.float32, copy=False)
        for start in range(0, n, fft_batch):
            stop = min(start + fft_batch, n)
            line_mask = mask[0]
            k_full = np.fft.fftshift(
                np.fft.fft(self.imgs[start:stop], axis=-1), axes=-1
            )
            zf_complex = np.fft.ifft(
                np.fft.ifftshift(k_full * line_mask, axes=-1), axis=-1
            )
            self.zf[start:stop] = np.abs(
                zf_complex
            ).astype(np.float32)
            self.k_meas[start:stop, 0] = zf_complex.real.astype(np.float32)
            self.k_meas[start:stop, 1] = zf_complex.imag.astype(np.float32)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, i: int):
        gt = torch.from_numpy(self.imgs[i])[None]
        if self.zf is None:
            return gt
        return (
            torch.from_numpy(self.zf[i])[None],
            torch.from_numpy(self.k_meas[i]),
            gt,
        )


def _agg(vals, cases):
    if not vals or len(vals) != len(cases):
        raise ValueError("metrics and case IDs must be nonempty and equally sized")
    v = np.asarray(vals, dtype=np.float64)
    per_case = {}
    for x, c in zip(v, cases):
        per_case.setdefault(c, []).append(x)
    cm = np.array([np.mean(per_case[c]) for c in sorted(per_case)])
    return {
        "slice_mean": float(v.mean()),
        "slice_std": float(v.std()),
        "n_slices": int(v.size),
        "case_mean": float(cm.mean()),
        "case_std": float(cm.std()),
        "n_cases": int(cm.size),
        "per_case": {c: float(np.mean(per_case[c])) for c in sorted(per_case)},
    }


@torch.no_grad()
def evaluate(model, loader, mask_t, device, case_of, final_hard_dc=True):
    """Evaluate full-image and ROI metrics with slice and case aggregation."""
    model.eval()
    acc = {k: [] for k in ("psnr", "ssim", "psnr_roi", "ssim_roi")}
    cases = []
    offset = 0
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 3:
            zf, k_meas, gt = batch
            zf = zf.to(device)
            k_meas = k_meas.to(device)
            gt = gt.to(device)
        else:
            gt = batch.to(device)
            zf, k_meas = simulate_cartesian_measurement(gt, mask_t)
        pred, _ = model(
            zf, mask_t, k_meas,
            final_hard_dc=final_hard_dc,
            fast_cartesian=True,
        )
        pred_np = pred.clamp(0, 1).cpu().numpy()[:, 0]
        gt_np = gt.cpu().numpy()[:, 0]
        for j, g in enumerate(gt_np):
            p = pred_np[j]
            bm = body_mask(g)
            acc["psnr"].append(psnr(g, p))
            acc["ssim"].append(ssim(g, p))
            acc["psnr_roi"].append(psnr_masked(g, p, bm))
            acc["ssim_roi"].append(ssim_masked(g, p, bm))
        batch_n = len(gt_np)
        cases.extend(case_of[offset:offset + batch_n])
        offset += batch_n
    if offset != len(case_of):
        raise ValueError("evaluation loader must cover all cases in order")

    full = {"psnr": _agg(acc["psnr"], cases), "ssim": _agg(acc["ssim"], cases)}
    roi = {"psnr": _agg(acc["psnr_roi"], cases), "ssim": _agg(acc["ssim_roi"], cases)}
    return {
        "psnr": full["psnr"]["slice_mean"],
        "psnr_std": full["psnr"]["slice_std"],
        "ssim": full["ssim"]["slice_mean"],
        "ssim_std": full["ssim"]["slice_std"],
        "n": full["psnr"]["n_slices"],
        "calibers": {"full_image": full, "roi_body": roi},
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable in this Python environment. "
            "Use --device cpu or run from a Python build with MPS enabled."
        )
    return torch.device(name)


def _checkpoint_name(mode: str, ratio: float) -> str:
    prefix = "ista" if mode == "fixed" else "ista_learnable"
    return f"{prefix}_cs{int(round(ratio * 100))}.pt"


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def experiment_config(args, ratio, meta):
    fields = (
        "mask_mode", "phases", "filters", "epochs", "batch", "lr", "mask_lr",
        "sym_w", "mask_gamma", "acs_frac", "seed", "mask_seed", "final_hard_dc",
    )
    return {
        **{k: getattr(args, k) for k in fields},
        "ratio": ratio, "size": meta["size"], "split": meta["split"],
        "schema_version": 2,
    }


def provenance(slice_dir):
    source = Path(__file__).parent
    paths = [source / name for name in ("ista_net.py", "train_ista.py", "sampling.py")]
    paths += [slice_dir / "meta.json", *sorted(slice_dir.glob("*.npz"))]
    checksums = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        checksums[path.name] = digest.hexdigest()
    return {
        "sha256": checksums,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scipy", "scikit-image")
        },
    }


def run_ratio(ratio, slice_dir, meta, args, device):
    size = int(meta["size"])
    config = experiment_config(args, ratio, meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"cs{int(round(ratio * 100))}"
    state_path = out_dir / f"ista_{args.mask_mode}_{tag}_state.pt"
    result_path = out_dir / f"ista_{args.mask_mode}_{tag}_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if not args.resume or result.get("config") != config:
            raise ValueError(f"existing result/config conflict: {result_path}")
        print(f"  completed {args.mask_mode} {tag}, reusing result", flush=True)
        return result
    if state_path.exists() and not args.resume:
        raise ValueError(f"use --resume or a fresh output directory: {state_path}")
    fingerprint = provenance(slice_dir)
    fixed_np = cartesian_mask(size, ratio, acs_frac=args.acs_frac, seed=args.mask_seed)
    fixed_t = torch.from_numpy(fixed_np).to(device)

    cache_mask = fixed_np if args.mask_mode == "fixed" else None
    tr = ReconSet(slice_dir, meta["split"]["train"], mask=cache_mask)
    te = ReconSet(slice_dir, meta["split"]["test"], mask=cache_mask)
    if set(meta["split"]["train"]) & set(meta["split"]["test"]):
        raise ValueError("train/test patient overlap")
    if len(tr) < args.batch:
        raise ValueError("training data must include at least one full batch")
    generator = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(
        tr,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )
    vl = DataLoader(te, batch_size=args.batch, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)
    model = ISTANetPlus(
        num_phases=args.phases,
        num_filters=args.filters,
    ).to(device)
    mask_module: Optional[LearnableMask] = None
    if args.mask_mode == "learnable":
        mask_module = LearnableMask(
            size=size,
            ratio=ratio,
            init_mask=fixed_t,
            acs_frac=args.acs_frac,
        ).to(device)

    params = [{"params": model.parameters(), "lr": args.lr}]
    if mask_module is not None:
        params.append({
            "params": mask_module.parameters(),
            "lr": args.mask_lr if args.mask_lr is not None else args.lr,
        })
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.L1Loss()

    hist = []
    records = []
    migrations = []
    started = datetime.now(timezone.utc).isoformat()
    start_epoch = 0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state.get("config") != config or state.get("provenance") != fingerprint:
            raise ValueError("resume requires identical configuration, data, source and environment")
        model.load_state_dict(state["model"])
        if mask_module is not None and state.get("mask") is not None:
            mask_module.load_state_dict(state["mask"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        hist = list(state.get("loss_hist", []))
        records = state["history"]
        migrations = state.get("migrations", [])
        if state.get("provenance_migration"):
            migrations.append(state["provenance_migration"])
        started = state["started_at"]
        generator.set_state(state["loader_rng"])
        torch.set_rng_state(state["torch_rng"])
        start_epoch = int(state["epoch"]) + 1
        print(
            f"  resume {args.mask_mode} {tag} from epoch {start_epoch}/{args.epochs}",
            flush=True,
        )

    for ep in range(start_epoch, args.epochs):
        model.train()
        if mask_module is not None:
            mask_module.train()
        t0, tot, n = time.time(), 0.0, 0
        rate_tot = 0.0
        l1_tot, sym_tot = 0.0, 0.0
        learning_rates = [group["lr"] for group in opt.param_groups]
        for batch in tl:
            if mask_module is None:
                zf, k_meas, gt = batch
                zf = zf.to(device)
                k_meas = k_meas.to(device)
                gt = gt.to(device)
                mask_t = fixed_t
            else:
                gt = batch.to(device)
                mask_t = mask_module()
                zf, k_meas = simulate_cartesian_measurement(gt, mask_t)
            opt.zero_grad(set_to_none=True)
            pred, sym = model(
                zf,
                mask_t,
                k_meas,
                final_hard_dc=args.final_hard_dc,
                fast_cartesian=True,
            )
            rate_penalty = (
                mask_module.rate_penalty()
                if mask_module is not None
                else pred.new_zeros(())
            )
            recon_loss = lossf(pred, gt)
            loss = recon_loss + args.sym_w * sym + args.mask_gamma * rate_penalty
            if not torch.isfinite(loss).item():
                raise FloatingPointError("nonfinite training loss; last epoch remains recoverable")
            loss.backward()
            opt.step()
            batch_n = len(gt)
            tot += loss.item() * batch_n
            rate_tot += float(rate_penalty.detach()) * batch_n
            l1_tot += float(recon_loss.detach()) * batch_n
            sym_tot += float(sym.detach()) * batch_n
            n += batch_n
        sched.step()
        hist.append(tot / max(n, 1))
        if mask_module is not None:
            with torch.no_grad():
                soft_rate = float(mask_module.probabilities().mean())
                hard_rate = float(mask_module.hard_mask().mean())
            mask_msg = f" soft_rate={soft_rate:.4f} hard_rate={hard_rate:.4f}"
        else:
            mask_msg = ""
        record = {
            "epoch": ep + 1, "loss": hist[-1], "l1": l1_tot / n,
            "symmetry": sym_tot / n, "rate_penalty": rate_tot / n,
            "seconds": time.time() - t0, "lr": learning_rates,
        }
        if mask_module is not None:
            record["mask"] = mask_module.summary()
        records.append(record)
        print(
            f"  {args.mask_mode} CS{int(ratio * 100):02d} "
            f"ep{ep + 1:02d}/{args.epochs} "
            f"L1+reg={hist[-1]:.5f} rate_pen={rate_tot / max(n, 1):.6f} "
            f"({time.time() - t0:.0f}s){mask_msg}",
            flush=True,
        )
        atomic_checkpoint(
            state_path,
            {
                "epoch": ep,
                "model": model.state_dict(),
                "mask": mask_module.state_dict() if mask_module is not None else None,
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "loss_hist": hist,
                "history": records,
                "config": config,
                "provenance": fingerprint,
                "torch_rng": torch.get_rng_state(),
                "loader_rng": generator.get_state(),
                "started_at": started,
                "migrations": migrations,
            },
        )
        atomic_json(state_path.with_suffix(".json"), {
            "status": "training", "completed_epochs": ep + 1,
            "config": config, "history": records, "started_at": started,
        })

    eval_mask = mask_module.hard_mask() if mask_module is not None else fixed_t
    res = evaluate(model, vl, eval_mask, device, te.case_of, args.final_hard_dc)
    ckpt = out_dir / _checkpoint_name(args.mask_mode, ratio)
    payload = {
        "model": model.state_dict(),
        "mask_mode": args.mask_mode,
        "ratio": ratio,
        "mask": mask_module.state_dict() if mask_module is not None else None,
        "final_hard_dc": args.final_hard_dc,
        "config": config,
        "provenance": fingerprint,
        "hard_mask": eval_mask.detach().cpu(),
        "migrations": migrations,
    }
    atomic_checkpoint(ckpt, payload)

    mask_summary = {
        "target_ratio": float(ratio),
        "target_lines": int(fixed_np[0].sum()),
        "hard_ratio": float(eval_mask[0].mean()),
        "hard_lines": int(eval_mask[0].sum()),
        "acs_lines": int(round(fixed_np[0].sum() * args.acs_frac)),
        "selected_lines": [int(x) for x in np.where(eval_mask[0].cpu().numpy() > 0.5)[0]],
        "line_probabilities": [float(x) for x in eval_mask[0].detach().cpu().numpy()],
    }
    if mask_module is not None:
        mask_summary = mask_module.summary()

    result = {
        "schema_version": 2,
        "status": "complete",
        "config": config,
        "provenance": fingerprint,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "migrations": migrations,
        "cs_ratio_nominal": ratio,
        "cs_ratio_actual": float(fixed_np[0].mean()),
        "sampled_lines": int(fixed_np[0].sum()),
        "size": size,
        "params_M": count_params(model) / 1e6,
        "phases": args.phases,
        "filters": args.filters,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch": args.batch,
        "sym_w": args.sym_w,
        "mask_mode": args.mask_mode,
        "mask_gamma": args.mask_gamma,
        "mask_lr": args.mask_lr if args.mask_lr is not None else args.lr,
        "final_hard_dc": args.final_hard_dc,
        "train_slices": len(tr),
        "test_slices": len(te),
        "checkpoint": ckpt.name,
        "loss_hist": hist,
        "history": records,
        "mask": mask_summary,
        "metrics": res,
    }
    atomic_json(result_path, result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/slices")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ratios", default="0.1,0.2,0.3")
    ap.add_argument("--mask-mode", choices=("fixed", "learnable"), default="fixed")
    ap.add_argument("--phases", type=int, default=9)
    ap.add_argument("--filters", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mask-lr", type=float, default=1e-3)
    ap.add_argument("--sym-w", type=float, default=0.01)
    ap.add_argument("--mask-gamma", type=float, default=10.0)
    ap.add_argument("--acs-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume each ratio from its per-ratio state file when present",
    )
    ap.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    ap.add_argument(
        "--no-final-hard-dc",
        dest="final_hard_dc",
        action="store_false",
        help="disable the final hard data-consistency projection",
    )
    ap.set_defaults(final_hard_dc=True)
    args = ap.parse_args()
    if min(args.epochs, args.batch, args.phases, args.filters) < 1:
        ap.error("epochs, batch, phases and filters must be positive")
    ratios = [float(x) for x in args.ratios.split(",")]
    if any(not 0 < r <= 1 for r in ratios):
        ap.error("ratios must be in (0, 1]")
    if not 0 < args.acs_frac <= 1 or min(args.lr, args.mask_lr) <= 0:
        ap.error("invalid ACS fraction or learning rate")

    slice_dir = Path(args.slices)
    meta = json.loads((slice_dir / "meta.json").read_text())
    device = _device(args.device)
    print(
        f"device={device} mode={args.mask_mode} phases={args.phases} "
        f"filters={args.filters} final_hard_dc={args.final_hard_dc}",
        flush=True,
    )

    result_name = (
        "ista_results.json" if args.mask_mode == "fixed"
        else "ista_learnable_results.json"
    )
    summary_path = Path(args.out, result_name)
    all_res: Dict[str, dict] = (
        json.loads(summary_path.read_text()) if summary_path.exists() else {}
    )
    for ratio in ratios:
        tag = f"cs{int(round(ratio * 100))}"
        all_res[tag] = run_ratio(ratio, slice_dir, meta, args, device)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        atomic_json(summary_path, all_res)

    print(
        f"\n=== ISTA-Net+ ({args.mask_mode}) 结果 "
        f"（测试集 {meta['n_cases']['test']} 例 / "
        f"{meta['n_slices']['test']} 切片）===",
        flush=True,
    )
    print(
        f"{'CS':>5} {'PSNR(全图)':>12} {'SSIM(全图)':>12} "
        f"{'PSNR(ROI/例)':>14} {'SSIM(ROI/例)':>14}",
        flush=True,
    )
    for tag, value in all_res.items():
        metrics = value["metrics"]
        roi = metrics["calibers"]["roi_body"]
        print(
            f"{tag:>5} {metrics['psnr']:>12.2f} {metrics['ssim']:>12.4f} "
            f"{roi['psnr']['case_mean']:>10.2f}±{roi['psnr']['case_std']:<3.2f}  "
            f"{roi['ssim']['case_mean']:>10.4f}±{roi['ssim']['case_std']:<5.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
