"""出图与出表：采样 mask、loss 曲线、重建/分割定性对比图、指标 Markdown 表。

图内文字用英文，避免 matplotlib 默认字体缺 CJK 字形导致方框。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sampling import cartesian_mask, psnr, ssim  # noqa: E402
from train_recon import data_consistency  # noqa: E402
from unet import UNet  # noqa: E402

RES = Path("results")
FIG = RES / "figures"
SLICES = Path("data/slices")


def fig_masks(size=192):
    ratios = [0.1, 0.2, 0.3]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, r in zip(axes, ratios):
        m = cartesian_mask(size, r, seed=0)
        ax.imshow(m, cmap="gray", interpolation="nearest")
        ax.set_title(f"CS {int(r*100)}%  ({int(m[0].sum())}/{size} lines)", fontsize=10)
        ax.set_xlabel("phase encode"); ax.set_ylabel("readout")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Cartesian 1D variable-density undersampling masks", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "sampling_masks.png", dpi=150)
    plt.close(fig)


def fig_loss():
    recon = json.loads((RES / "recon_results.json").read_text())
    seg_p = RES / "seg_results.json"
    n = 2 if seg_p.exists() else 1
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.6), squeeze=False)
    ax = axes[0][0]
    for k, v in recon.items():
        ax.plot(range(1, len(v["loss_hist"]) + 1), v["loss_hist"],
                marker="o", ms=3, label=f"CS {k[2:]}%")
    ax.set_xlabel("epoch"); ax.set_ylabel("train L1"); ax.legend()
    ax.set_title("Reconstruction U-Net training loss"); ax.grid(alpha=.3)
    if n == 2:
        seg = json.loads(seg_p.read_text())
        ax2 = axes[0][1]
        for k, v in seg.items():
            ax2.plot(range(1, len(v["loss_hist"]) + 1), v["loss_hist"],
                     marker="o", ms=3, label=k)
        ax2.set_xlabel("epoch"); ax2.set_ylabel("train loss"); ax2.legend(fontsize=8)
        ax2.set_title("Segmentation loss (config A/B)"); ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "loss_curves.png", dpi=150)
    plt.close(fig)


def fig_recon_qualitative(device):
    meta = json.loads((SLICES / "meta.json").read_text())
    cid = meta["split"]["test"][0]
    d = np.load(SLICES / f"{cid}.npz")
    fg_idx = np.where(d["fg"])[0]
    gt = d["img"][fg_idx[len(fg_idx) // 2]]

    ratios = [0.1, 0.2, 0.3]
    fig, axes = plt.subplots(len(ratios), 4, figsize=(11, 2.9 * len(ratios)))
    for row, r in enumerate(ratios):
        mask = cartesian_mask(gt.shape[0], r, seed=0)
        k = np.fft.fftshift(np.fft.fft2(gt)) * mask
        zf = np.abs(np.fft.ifft2(np.fft.ifftshift(k))).astype(np.float32)
        ck = RES / f"recon_cs{int(r*100)}.pt"
        model = UNet(1, 1, base=32, residual=True).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(zf)[None, None].to(device))
            pred = pred.clamp(0, 1).cpu().numpy()[0, 0]
        dc = np.clip(data_consistency(pred, k, mask), 0, 1)
        panels = [("ground truth (full k-space)", gt), (f"zero-filled CS{int(r*100)}%", zf),
                  ("U-Net", pred), ("U-Net + DC", dc)]
        for col, (title, im) in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            t = title if col == 0 else f"{title}\n{psnr(gt,im):.2f} dB / {ssim(gt,im):.4f}"
            ax.set_title(t, fontsize=9)
            ax.axis("off")
    fig.suptitle(f"Reconstruction, test case {cid} (PSNR / SSIM)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "recon_qualitative.png", dpi=150)
    plt.close(fig)


def fig_seg_qualitative(device):
    seg = json.loads((RES / "seg_results.json").read_text())
    meta = json.loads((SLICES / "meta.json").read_text())
    names = [n for n in ("baseline", "tuned") if (RES / f"seg_{n}.pt").exists()]
    cid = meta["split"]["test"][0]
    d = np.load(SLICES / f"{cid}.npz")
    fg_idx = np.where(d["fg"])[0]
    picks = fg_idx[[len(fg_idx) // 4, len(fg_idx) // 2, 3 * len(fg_idx) // 4]]

    ncol = 2 + len(names)
    fig, axes = plt.subplots(len(picks), ncol, figsize=(2.7 * ncol, 2.8 * len(picks)))
    for row, si in enumerate(picks):
        im, lb = d["img"][si], d["lab"][si]
        axes[row][0].imshow(im, cmap="gray"); axes[row][0].set_title("image", fontsize=9)
        axes[row][1].imshow(lb, cmap="magma"); axes[row][1].set_title("ground-truth mask", fontsize=9)
        for c, name in enumerate(names):
            model = UNet(1, 1, base=32, residual=False).to(device)
            model.load_state_dict(torch.load(RES / f"seg_{name}.pt", map_location=device))
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(im)[None, None].to(device)))
                p = (p.cpu().numpy()[0, 0] > 0.5).astype(np.float32)
            dsc = (2 * (p * lb).sum() + 1e-6) / (p.sum() + lb.sum() + 1e-6)
            ax = axes[row][2 + c]
            ax.imshow(p, cmap="magma")
            ax.set_title(f"{name}\nDice {dsc:.3f}", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"Left-atrium segmentation, test case {cid}", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "seg_qualitative.png", dpi=150)
    plt.close(fig)


def write_tables():
    lines = ["# 实验结果（真实测量值）\n"]
    meta = json.loads((SLICES / "meta.json").read_text())
    lines.append(f"数据集：MSD Task02_Heart（心脏 MRI，左心房分割，20 例 `.nii.gz`）\n")
    lines.append(f"划分：**按病例**划分，train {meta['n_cases']['train']} 例 / "
                 f"test {meta['n_cases']['test']} 例；切片 "
                 f"{meta['n_slices']['train']} / {meta['n_slices']['test']}；"
                 f"分辨率 {meta['size']}×{meta['size']}\n")

    recon = json.loads((RES / "recon_results.json").read_text())
    lines.append("\n## 实验1 K 空间欠采样重建（指标 PSNR / SSIM）\n")
    lines.append("| CS 比率 | 采样线数 | 零填充 PSNR | U-Net PSNR | U-Net+DC PSNR | "
                 "零填充 SSIM | U-Net SSIM | U-Net+DC SSIM |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for k, v in recon.items():
        m = v["metrics"]
        lines.append(
            f"| {k[2:]}% | {v['sampled_lines']}/{v['size']} | {m['zf']['psnr']:.2f} dB | "
            f"**{m['unet']['psnr']:.2f} dB** | **{m['unet_dc']['psnr']:.2f} dB** | "
            f"{m['zf']['ssim']:.4f} | **{m['unet']['ssim']:.4f}** | "
            f"**{m['unet_dc']['ssim']:.4f}** |")
    v0 = next(iter(recon.values()))
    lines.append(f"\n训练配置：U-Net {v0['params_M']:.2f}M 参数，残差输出，L1 损失，"
                 f"Adam lr {v0['lr']}，cosine 退火，{v0['epochs']} epoch，batch {v0['batch']}。")

    seg_p = RES / "seg_results.json"
    if seg_p.exists():
        seg = json.loads(seg_p.read_text())
        lines.append("\n## 实验2 左心房分割 损失/优化器/增强 A-B 对照（指标 Dice）\n")
        lines.append("| 配置 | 损失 | 优化器 | LR 策略 | 增强 | Dice(按病例) | Dice(前景切片) |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, v in seg.items():
            c, m = v["config"], v["metrics"]
            lines.append(
                f"| {k} | {c['loss']} | {c['optim']} | "
                f"{'cosine' if c['cosine'] else 'const'} | "
                f"{'flip+rot90' if c['augment'] else '-'} | "
                f"**{m['dice_case_mean']:.4f}** ± {m['dice_case_std']:.4f} | "
                f"{m['dice_slice_mean_fg']:.4f} |")
        base = seg.get("baseline", {}).get("metrics", {}).get("dice_case_mean")
        tun = seg.get("tuned", {}).get("metrics", {}).get("dice_case_mean")
        if base and tun:
            lines.append(f"\n调优版相对 baseline：Dice {base:.4f} → {tun:.4f}"
                         f"（绝对 +{tun-base:.4f}，相对 +{(tun-base)/base*100:.1f}%）。")

    lines.append("\n## 方法学边界（面试可直接引用）\n")
    lines.append("- MSD 数据是**已重建的幅值图**，不是原始多线圈 k 空间。这里做的是 "
                 "retrospective undersampling（幅值图 FFT 后抽样），没有真实相位与线圈灵敏度信息，"
                 "绝对 PSNR 偏乐观，**不能与 fastMRI 上的论文数字直接比较**。")
    lines.append("- 划分在**病例级**而非切片级。同一病例相邻切片高度相似，"
                 "按切片随机划分会造成测试集泄漏、指标虚高。")
    lines.append("- 欠采样用笛卡尔一维变密度随机（只抽相位编码线，中心保留 ACS），"
                 "对应临床 2D 序列真正能省的时间；二维随机点采样在 2D 序列里物理不可实现。")
    lines.append("- 单次运行、单一 seed，未做多 seed 重复，Dice/PSNR 的置信区间未估计。")
    (RES / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fig_masks()
    if (RES / "recon_results.json").exists():
        fig_loss()
        fig_recon_qualitative(device)
    if (RES / "seg_results.json").exists():
        fig_seg_qualitative(device)
    write_tables()
