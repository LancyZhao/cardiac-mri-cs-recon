"""用已训练好的 checkpoint 重算重建指标，不重新训练。

存在的理由：results/recon_results.json 里报的是**全图口径 + 按切片聚合**，
这两个选择都让绝对值偏乐观、且低估方差：

- 全图口径：测试集 670 张切片里像素 <0.05 的近零背景平均占 51.5%，GT 与重建
  在那里都≈0、误差≈0，把 PSNR=10log10(1/MSE) 的分母稀释掉了；SSIM 在均匀零区
  的局部窗口≈1，被抬得更多。
- 按切片聚合：670 张相邻切片高度相关，不是 670 个独立样本，切片间 std 衡量的是
  切片难易度差异，不是"换一批病人会怎样"。分割实验（train_seg.py）已经按病例
  聚合并报例间 std，重建实验当时没做 —— 同一项目里两个口径不一致。

本脚本对同一批 checkpoint 同时给出 full_image / roi_body 两种口径 × slice / case
两种聚合，写入 results/recon_metrics_calibers.json。已公开的数字对应
full_image + slice，可原样复现，不被覆盖。

用法：
    python src/eval_recon.py                      # 读 results/recon_cs{10,20,30}.pt
    python src/eval_recon.py --ratios 0.3         # 只算 CS30
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from sampling import cartesian_mask  # noqa: E402
from train_recon import ReconSet, evaluate  # noqa: E402
from unet import UNet, count_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/slices")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ratios", default="0.1,0.2,0.3")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--mask-seed", type=int, default=0)
    args = ap.parse_args()

    slice_dir, out_dir = Path(args.slices), Path(args.out)
    meta = json.loads((slice_dir / "meta.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    size = meta["size"]
    print(f"device={device} | test cases={meta['n_cases']['test']} "
          f"slices={meta['n_slices']['test']}", flush=True)

    all_res = {}
    for ratio in [float(x) for x in args.ratios.split(",")]:
        tag = f"cs{int(ratio * 100)}"
        ckpt = out_dir / f"recon_{tag}.pt"
        if not ckpt.exists():
            print(f"[skip] {ckpt} 不存在——先跑 train_recon.py", flush=True)
            continue

        # mask 必须和训练时同一个 seed，否则评估的是另一个采样模式
        mask = cartesian_mask(size, ratio, seed=args.mask_seed)
        te = ReconSet(slice_dir, meta["split"]["test"], mask)
        vl = DataLoader(te, batch_size=args.batch, shuffle=False, num_workers=0)

        model = UNet(1, 1, base=args.base, residual=True).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"[{tag}] loaded {ckpt.name} ({count_params(model)/1e6:.2f}M params), "
              f"{int(mask[0].sum())}/{size} lines", flush=True)

        all_res[tag] = {
            "cs_ratio_nominal": ratio,
            "cs_ratio_actual": float(mask[0].mean()),
            "sampled_lines": int(mask[0].sum()),
            "checkpoint": ckpt.name,
            "metrics": evaluate(model, vl, mask, device, case_of=te.case_of),
        }
        (out_dir / "recon_metrics_calibers.json").write_text(
            json.dumps(all_res, indent=2, ensure_ascii=False))

    if not all_res:
        return
    print("\n=== 口径对照（PSNR dB）===")
    hdr = f"{'CS':>5} {'方法':>9} {'全图/切片':>12} {'全图/病例':>16} {'ROI/病例':>16}"
    print(hdr)
    for tag, v in all_res.items():
        for m in ("zf", "unet", "unet_dc"):
            c = v["metrics"][m]["calibers"]
            f_s, f_c, r_c = c["full_image"]["psnr"], c["full_image"]["psnr"], c["roi_body"]["psnr"]
            print(f"{tag:>5} {m:>9} {f_s['slice_mean']:>12.2f} "
                  f"{f_c['case_mean']:>10.2f}±{f_c['case_std']:<5.2f} "
                  f"{r_c['case_mean']:>10.2f}±{r_c['case_std']:<5.2f}")
    print(f"\n→ {out_dir / 'recon_metrics_calibers.json'}")


if __name__ == "__main__":
    main()
