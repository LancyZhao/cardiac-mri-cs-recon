"""Validate measured experiments and export thesis tables and research figures."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ista_net import ISTANetPlus, simulate_cartesian_measurement
from sampling import psnr, ssim
from train_ista import ReconSet, _device, atomic_json, evaluate
from train_recon import data_consistency
from unet import UNet

TAGS = ("cs10", "cs20", "cs30")


def load(path):
    return json.loads(Path(path).read_text())


def validate(entry, folder):
    if entry.get("schema_version") != 2 or entry.get("status") != "complete":
        raise ValueError(f"unverified or incomplete experiment in {folder}")
    config = entry["config"]
    if len(entry["history"]) != config["epochs"]:
        raise ValueError("incomplete epoch history")
    if entry["metrics"]["n"] != entry["test_slices"]:
        raise ValueError("incomplete evaluation")
    if not np.isfinite([entry["metrics"]["psnr"], entry["metrics"]["ssim"]]).all():
        raise ValueError("nonfinite metrics")
    mask = entry["mask"]
    count = round(config["size"] * config["ratio"])
    if mask["hard_lines"] != count or len(set(mask["selected_lines"])) != count:
        raise ValueError("sampling budget mismatch")
    n_acs = max(1, round(count * config["acs_frac"]))
    lo = config["size"] // 2 - n_acs // 2
    if not set(range(lo, lo + n_acs)).issubset(mask["selected_lines"]):
        raise ValueError("ACS lines are missing")
    checkpoint = Path(folder) / entry["checkpoint"]
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["config"] != config or payload["provenance"] != entry["provenance"]:
        raise ValueError("checkpoint/result provenance mismatch")
    expected = torch.zeros(config["size"], config["size"])
    expected[:, mask["selected_lines"]] = 1
    torch.testing.assert_close(payload["hard_mask"], expected, rtol=0, atol=0)
    return payload


def load_model(entry, folder, device):
    payload = validate(entry, folder)
    model = ISTANetPlus(entry["phases"], entry["filters"]).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload["hard_mask"].to(device)


def markdown_table(header, rows):
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *["| " + " | ".join(map(str, row)) + " |" for row in rows],
    ])


def collect(root, baseline, partial=False):
    data = {"fixed": {}, "learnable": {}, "phases": {}, "baseline": load(
        baseline / "recon_metrics_calibers.json")}
    missing = []
    for mode, name in (("fixed", "ista_results.json"),
                       ("learnable", "ista_learnable_results.json")):
        values = load(root / name) if (root / name).exists() else {}
        for tag in TAGS:
            if tag not in values:
                missing.append(f"{mode}/{tag}")
                continue
            validate(values[tag], root)
            data[mode][tag] = values[tag]
    for k in (3, 6, 9, 12):
        path = root / "ablation" / f"phases_k{k}" / "ista_results.json"
        entry = data["fixed"].get("cs20") if k == 9 else (
            load(path).get("cs20") if path.exists() else None)
        if entry is None:
            missing.append(f"phases/{k}")
        else:
            if k != 9:
                validate(entry, path.parent)
            data["phases"][str(k)] = entry
    reference = data["fixed"].get("cs20")
    if reference:
        same = ("filters", "epochs", "batch", "lr", "sym_w",
                "acs_frac", "seed", "mask_seed", "split", "final_hard_dc")
        for mode in ("fixed", "learnable", "phases"):
            for entry in data[mode].values():
                for key in same:
                    if entry["config"][key] != reference["config"][key]:
                        raise ValueError(f"unmatched experiment setting: {mode}/{key}")
    if missing and not partial:
        raise ValueError("experiments are not complete: " + ", ".join(missing))
    data["missing"] = missing
    data["status"] = "partial" if missing else "complete"
    return data


def tables(data):
    methods = [
        ("零填充IFFT", "baseline", "zf"),
        ("U-Net", "baseline", "unet"),
        ("U-Net+DC", "baseline", "unet_dc"),
        ("ISTA-Net+（固定mask）", "fixed", None),
        ("ISTA-Net+（可学习mask）", "learnable", None),
    ]
    exports = {}
    for key in ("psnr", "ssim", "roi"):
        rows = []
        for label, group, sub in methods:
            row = [label]
            for tag in TAGS:
                if tag not in data[group]:
                    row.append("尚未完成")
                    continue
                metrics = data[group][tag]["metrics"]
                if sub:
                    metrics = metrics[sub]
                if key == "roi":
                    stat = metrics["calibers"]["roi_body"]["psnr"]
                    row.append(f"{stat['case_mean']:.2f}±{stat['case_std']:.2f}")
                else:
                    row.append(format(metrics[key], ".2f" if key == "psnr" else ".4f"))
            rows.append(row)
        exports[f"{key}_table"] = markdown_table(
            ["方法", "CS 10%", "CS 20%", "CS 30%"], rows)
    rows = []
    for k in (3, 6, 9, 12):
        entry = data["phases"].get(str(k))
        rows.append([k, f"{entry['metrics']['psnr']:.2f}" if entry else "尚未完成",
                     f"{entry['metrics']['ssim']:.4f}" if entry else "尚未完成",
                     f"{37442*k/1e6:.6f}"])
    exports["phase_table"] = markdown_table(
        ["K", "PSNR（dB）", "SSIM", "参数量（M）"], rows)
    return exports


@torch.no_grad()
def hard_dc_ablation(data, root, slices, device):
    entry = data["fixed"]["cs20"]
    model, mask = load_model(entry, root, device)
    meta = load(slices / "meta.json")
    ds = ReconSet(slices, meta["split"]["test"], mask=mask.cpu().numpy())
    dl = DataLoader(ds, batch_size=entry["batch"], shuffle=False)
    without = evaluate(model, dl, mask, device, ds.case_of, final_hard_dc=False)
    return {
        "design": "same trained K=9 CS20 checkpoint, inference projection switch only",
        "with_hard_dc": entry["metrics"], "without_hard_dc": without,
        "checkpoint": entry["checkpoint"], "config": entry["config"],
    }


def save_figure(fig, path):
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_curves(data, figures):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    for ax, tag in zip(axes, TAGS):
        for mode, color in (("fixed", "#2366a5"), ("learnable", "#c64b48")):
            entry = data[mode][tag]
            ax.plot([x["epoch"] for x in entry["history"]],
                    [x["l1"] for x in entry["history"]], label=mode, color=color)
        ax.set(xlabel="Epoch", ylabel="Training L1", title=f"CS {tag[2:]}%")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    save_figure(fig, figures / "ista_loss.png")


def plot_masks(data, figures):
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    details = {}
    for row, tag in enumerate(TAGS):
        f = data["fixed"][tag]
        l = data["learnable"][tag]
        size = f["size"]
        lines = []
        for col, entry in enumerate((f, l)):
            indices = entry["mask"]["selected_lines"]
            pattern = np.zeros((size, size))
            pattern[:, indices] = 1
            axes[row, col].imshow(pattern, cmap="gray", vmin=0, vmax=1, aspect="auto")
            axes[row, col].set(title=f"{('Fixed','Learned')[col]} CS {tag[2:]}%: {len(indices)} lines",
                               xlabel="Phase-encoding index")
            axes[row, col].set_yticks([])
            lines.append(set(indices))
        ax = axes[row, 2]
        ax.plot(l["mask"]["line_probabilities"], color="#c64b48")
        ax.scatter(sorted(lines[1]), [1]*len(lines[1]), marker="|", s=30, color="#222222")
        ax.set(xlabel="Phase-encoding index", ylabel="Surrogate probability",
               title="Learned probabilities and hard support", ylim=(-0.03, 1.06))
        n_acs = l["mask"]["acs_lines"]
        lo = size//2-n_acs//2
        ax.axvspan(lo, lo+n_acs-1, color="#259380", alpha=0.2)
        details[tag] = {
            "changed_lines": len(lines[1] - lines[0]),
            "fixed_mean_abs_frequency": float(np.mean(np.abs(np.array(sorted(lines[0]))-size//2))),
            "learned_mean_abs_frequency": float(np.mean(np.abs(np.array(sorted(lines[1]))-size//2))),
            "fixed_conjugate_union": len(lines[0] | {(-i) % size for i in lines[0]}),
            "learned_conjugate_union": len(lines[1] | {(-i) % size for i in lines[1]}),
        }
    fig.tight_layout()
    save_figure(fig, figures / "ista_masks.png")
    return details


@torch.no_grad()
def plot_recon(data, root, baseline, slices, figures, device):
    meta = load(slices / "meta.json")
    cid = meta["split"]["test"][0]
    with np.load(slices / f"{cid}.npz") as case:
        foreground = np.flatnonzero(case["fg"])
        index = int(foreground[len(foreground)//2])
        gt = case["img"][index].copy()
    x = torch.from_numpy(gt)[None, None].to(device)
    selection = {"case": cid, "slice_index": index,
                 "rule": "first test case, middle foreground slice, selected without model scores"}
    for tag in TAGS:
        fixed, mask_f = load_model(data["fixed"][tag], root, device)
        learned, mask_l = load_model(data["learnable"][tag], root, device)
        zf_f, kf = simulate_cartesian_measurement(x, mask_f)
        zf_l, kl = simulate_cartesian_measurement(x, mask_l)
        pred_f, _ = fixed(zf_f, mask_f, kf, fast_cartesian=True)
        pred_l, _ = learned(zf_l, mask_l, kl, fast_cartesian=True)
        unet = UNet(1, 1, base=32, residual=True).to(device)
        unet.load_state_dict(torch.load(baseline / f"recon_{tag}.pt", map_location=device, weights_only=True))
        unet.eval()
        pu = unet(zf_f).clamp(0, 1).cpu().numpy()[0, 0]
        kf_full = np.fft.fftshift(np.fft.fft2(gt)) * mask_f.cpu().numpy()
        pu = data_consistency(pu, kf_full, mask_f.cpu().numpy()).clip(0, 1)
        panels = [
            ("Reference", gt), ("ZF fixed", zf_f.cpu().numpy()[0, 0]),
            ("U-Net + DC", pu), ("ISTA fixed", pred_f.cpu().numpy()[0, 0]),
            ("ZF learned", zf_l.cpu().numpy()[0, 0]),
            ("ISTA learned", pred_l.cpu().numpy()[0, 0]),
        ]
        fig, axes = plt.subplots(4, 3, figsize=(9, 10.5), layout="constrained")
        for panel, (title, image) in enumerate(panels):
            row, col = 2 * (panel // 3), panel % 3
            axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
            caption = title if panel == 0 else f"{title}\n{psnr(gt,image):.2f} dB / {ssim(gt,image):.4f}"
            axes[row, col].set_title(caption, fontsize=11, pad=6)
            error = np.abs(image - gt)
            axes[row + 1, col].imshow(error, cmap="magma", vmin=0, vmax=0.2)
            axes[row + 1, col].set_title("Absolute error [0, 0.2]", fontsize=10, pad=6)
            for ax in axes[row:row + 2, col]:
                ax.axis("off")
        fig.suptitle(f"{cid}, slice {index}, CS {tag[2:]}%")
        save_figure(fig, figures / f"ista_recon_{tag}.png")
    return selection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, default=Path("results"))
    ap.add_argument("--slices", type=Path, default=Path("data/slices"))
    ap.add_argument("--device", default="cpu", choices=("auto", "cpu", "mps"))
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--reuse-dc", action="store_true",
                    help="reuse a previously evaluated DC ablation when regenerating figures")
    args = ap.parse_args()
    data = collect(args.results, args.baseline, args.partial)
    exports = tables(data)
    if not data["missing"]:
        device = _device(args.device)
        dc_path = args.results / "hard_dc_ablation.json"
        dc = load(dc_path) if args.reuse_dc else hard_dc_ablation(
            data, args.results, args.slices, device)
        if dc["config"] != data["fixed"]["cs20"]["config"]:
            raise ValueError("DC ablation configuration mismatch")
        atomic_json(args.results / "hard_dc_ablation.json", dc)
        data["hard_dc"] = dc
        exports["dc_table"] = markdown_table(
            ["推理配置（同一checkpoint）", "PSNR（dB）", "SSIM"],
            [[label, f"{dc[k]['psnr']:.2f}", f"{dc[k]['ssim']:.4f}"]
             for label, k in (("关闭末尾硬DC", "without_hard_dc"),
                              ("开启末尾硬DC", "with_hard_dc"))])
        fig_dir = args.results / "figures"
        fig_dir.mkdir(exist_ok=True)
        plot_curves(data, fig_dir)
        data["mask_analysis"] = plot_masks(data, fig_dir)
        data["qualitative_selection"] = plot_recon(
            data, args.results, args.baseline, args.slices, fig_dir, device)
    else:
        exports["dc_table"] = "推理投影消融尚未完成。"
    data["tables"] = exports
    atomic_json(args.results / "thesis_results.json", data)
    report = ["# ISTA-Net+ 实验报告", "",
              f"状态：{data['status']}。仅使用实测结果；缺失结果不作估计。", ""]
    for title, key in (
        ("全图PSNR（按切片）", "psnr_table"),
        ("全图SSIM（按切片）", "ssim_table"),
        ("体内ROI PSNR（按病例，均值±总体标准差）", "roi_table"),
        ("独立训练的Phase数消融（CS20）", "phase_table"),
        ("末尾DC投影消融（CS20）", "dc_table"),
    ):
        report.extend(["## " + title, "", exports[key], ""])
    report.extend(["U-Net基线20轮，ISTA-Net+各配置50轮；主比较并非等训练预算。",
                   "历史调试阶段已使用本测试划分，故结果属于探索性比较；需要新的病例验证泛化。",
                   "mask、网络均只用训练病例更新；最终评估使用末轮checkpoint，不按测试分数挑选模型。"])
    (args.results / "REPORT.md").write_text("\n".join(report) + "\n")
    print(f"report status={data['status']}, missing={data['missing']}", flush=True)


if __name__ == "__main__":
    main()
