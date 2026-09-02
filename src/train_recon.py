"""实验1：K 空间欠采样重建。CS 10/20/30% 下比较 零填充 / U-Net / U-Net+数据一致性。

评价指标 PSNR/SSIM，对应申报书第 313 行立的 KPI（CS 10/20/30% -> 31.12/37.35/39.93 dB）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from sampling import (  # noqa: E402
    body_mask, cartesian_mask, psnr, psnr_masked, ssim, ssim_masked,
)
from unet import UNet, count_params  # noqa: E402


class ReconSet(Dataset):
    """按病例列表加载切片；mask 固定，欠采样在 __getitem__ 里即时算。

    每张切片记住它来自哪个病例（self.case_of[i]），供按病例聚合指标使用——
    670 张相邻切片高度相关，不是 670 个独立样本，切片间 std 会低估真实方差。
    """

    def __init__(self, slice_dir: Path, case_ids, mask: np.ndarray):
        self.mask = mask
        self.imgs = []
        self.case_of = []
        for cid in case_ids:
            d = np.load(slice_dir / f"{cid}.npz")
            self.imgs.append(d["img"])
            self.case_of.extend([cid] * len(d["img"]))
        self.imgs = np.concatenate(self.imgs, axis=0)
        assert len(self.case_of) == len(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        gt = self.imgs[i]
        k = np.fft.fftshift(np.fft.fft2(gt)) * self.mask
        zf = np.abs(np.fft.ifft2(np.fft.ifftshift(k))).astype(np.float32)
        return torch.from_numpy(zf)[None], torch.from_numpy(gt)[None]


def data_consistency(pred: np.ndarray, zf_k: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把网络输出里"已被真实测量过"的 k 空间行换回测量值——重建里的硬先验。"""
    kp = np.fft.fftshift(np.fft.fft2(pred))
    kp = kp * (1 - mask) + zf_k * mask
    return np.abs(np.fft.ifft2(np.fft.ifftshift(kp))).astype(np.float32)


def _agg(vals, cases):
    """把逐切片值聚合成两种口径：按切片等权，和先按病例内平均再对病例平均。

    按病例聚合才是有效的不确定性估计——测试集是 6 例 670 张切片，切片间 std
    衡量的是"不同切片难易度差异"，不是"换一批病人会怎样"。
    """
    v = np.asarray(vals, dtype=np.float64)
    per_case = {}
    for x, c in zip(v, cases):
        per_case.setdefault(c, []).append(x)
    cm = np.array([np.mean(per_case[c]) for c in sorted(per_case)])
    return {
        "slice_mean": float(v.mean()), "slice_std": float(v.std()), "n_slices": int(v.size),
        "case_mean": float(cm.mean()), "case_std": float(cm.std()), "n_cases": int(cm.size),
        "per_case": {c: float(np.mean(per_case[c])) for c in sorted(per_case)},
    }


@torch.no_grad()
def evaluate(model, loader, mask, device, with_dc=True, case_of=None):
    """两种口径 × 两种聚合。

    口径：full_image（全图，含约 51.5% 近零背景，与已公开数字一致）
          roi_body （只在 GT 估出的体内区域内算，背景不再稀释 MSE）
    聚合：slice（670 张等权，旧口径）/ case（6 例，先例内平均）
    """
    model.eval()
    keys = ("psnr", "ssim", "psnr_roi", "ssim_roi")
    acc = {m: {k: [] for k in keys} for m in ("zf", "unet", "unet_dc")}
    cases, i = [], 0
    for zf, gt in loader:
        pred = model(zf.to(device)).clamp(0, 1).cpu().numpy()[:, 0]
        zf_np, gt_np = zf.numpy()[:, 0], gt.numpy()[:, 0]
        for j in range(len(gt_np)):
            g = gt_np[j]
            bm = body_mask(g)  # 只依赖 GT，不依赖被评估的重建结果
            variants = {"zf": zf_np[j], "unet": pred[j]}
            if with_dc:
                zf_k = np.fft.fftshift(np.fft.fft2(g)) * mask
                variants["unet_dc"] = np.clip(data_consistency(pred[j], zf_k, mask), 0, 1)
            for name, im in variants.items():
                acc[name]["psnr"].append(psnr(g, im))
                acc[name]["ssim"].append(ssim(g, im))
                acc[name]["psnr_roi"].append(psnr_masked(g, im, bm))
                acc[name]["ssim_roi"].append(ssim_masked(g, im, bm))
            cases.append(case_of[i] if case_of else "all")
            i += 1

    out = {}
    for m, d in acc.items():
        if not d["psnr"]:
            continue
        full = {"psnr": _agg(d["psnr"], cases), "ssim": _agg(d["ssim"], cases)}
        roi = {"psnr": _agg(d["psnr_roi"], cases), "ssim": _agg(d["ssim_roi"], cases)}
        out[m] = {
            # 旧字段保持不变：全图 + 按切片聚合，已公开的数字仍能原样对上
            "psnr": full["psnr"]["slice_mean"], "psnr_std": full["psnr"]["slice_std"],
            "ssim": full["ssim"]["slice_mean"], "ssim_std": full["ssim"]["slice_std"],
            "n": full["psnr"]["n_slices"],
            "calibers": {"full_image": full, "roi_body": roi},
        }
    return out


def run_ratio(ratio, slice_dir, meta, args, device):
    size = meta["size"]
    mask = cartesian_mask(size, ratio, seed=args.mask_seed)
    actual = float(mask[0].mean())
    tr = ReconSet(slice_dir, meta["split"]["train"], mask)
    te = ReconSet(slice_dir, meta["split"]["test"], mask)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(te, batch_size=args.batch, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)
    model = UNet(1, 1, base=args.base, residual=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.L1Loss()

    hist = []
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for zf, gt in tl:
            zf, gt = zf.to(device), gt.to(device)
            opt.zero_grad()
            loss = lossf(model(zf), gt)
            loss.backward()
            opt.step()
            tot += loss.item() * len(zf)
            n += len(zf)
        sched.step()
        hist.append(tot / max(n, 1))
        print(f"  CS{int(ratio*100):02d} ep{ep+1:02d}/{args.epochs} "
              f"L1={hist[-1]:.5f} ({time.time()-t0:.0f}s)", flush=True)

    res = evaluate(model, vl, mask, device, case_of=te.case_of)
    torch.save(model.state_dict(), Path(args.out) / f"recon_cs{int(ratio*100)}.pt")
    return {
        "cs_ratio_nominal": ratio, "cs_ratio_actual": actual,
        "sampled_lines": int(mask[0].sum()), "size": size,
        "params_M": count_params(model) / 1e6,
        "epochs": args.epochs, "lr": args.lr, "batch": args.batch, "loss": "L1",
        "train_slices": len(tr), "test_slices": len(te),
        "loss_hist": hist, "metrics": res,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/slices")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ratios", default="0.1,0.2,0.3")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-seed", type=int, default=0)
    args = ap.parse_args()

    slice_dir = Path(args.slices)
    meta = json.loads((slice_dir / "meta.json").read_text())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device} | train cases={meta['n_cases']['train']} "
          f"test cases={meta['n_cases']['test']}", flush=True)

    all_res = {}
    for r in [float(x) for x in args.ratios.split(",")]:
        all_res[f"cs{int(r*100)}"] = run_ratio(r, slice_dir, meta, args, device)
        Path(args.out, "recon_results.json").write_text(json.dumps(all_res, indent=2))

    print("\n=== 重建结果（测试集 %d 例 / %d 切片，按病例划分）===" %
          (meta["n_cases"]["test"], meta["n_slices"]["test"]))
    print("全图口径（含约 51.5% 近零背景，与已公开数字一致）")
    print(f"{'CS':>5} {'零填充 PSNR':>13} {'U-Net PSNR':>12} {'U-Net+DC':>10} "
          f"{'零填充 SSIM':>13} {'U-Net SSIM':>12}")
    for k, v in all_res.items():
        m = v["metrics"]
        print(f"{k:>5} {m['zf']['psnr']:>13.2f} {m['unet']['psnr']:>12.2f} "
              f"{m['unet_dc']['psnr']:>10.2f} {m['zf']['ssim']:>13.4f} "
              f"{m['unet']['ssim']:>12.4f}")
    print("\nROI(体内)口径，按病例聚合 ± 例间 std（审稿人常要求的口径）")
    print(f"{'CS':>5} {'零填充 PSNR':>18} {'U-Net+DC PSNR':>18} {'U-Net+DC SSIM':>18}")
    for k, v in all_res.items():
        r = v["metrics"]["unet_dc"]["calibers"]["roi_body"]
        z = v["metrics"]["zf"]["calibers"]["roi_body"]
        print(f"{k:>5} {z['psnr']['case_mean']:>11.2f}±{z['psnr']['case_std']:<5.2f} "
              f"{r['psnr']['case_mean']:>11.2f}±{r['psnr']['case_std']:<5.2f} "
              f"{r['ssim']['case_mean']:>11.4f}±{r['ssim']['case_std']:<5.4f}")


if __name__ == "__main__":
    main()
