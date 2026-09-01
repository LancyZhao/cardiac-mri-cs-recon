"""数据流水线：.nii.gz 心脏 MRI -> 归一化 2D 切片数据集（按病例划分训练/测试）。

对应中期表里的"读取、格式转换、尺寸调整、归一化，划分训练集与测试集"。
关键方法学取舍：按 **病例(patient)** 划分而不是按切片划分——同一病例的相邻切片
高度相似，按切片随机划分会让测试集泄漏进训练集，指标虚高。
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.transform import resize


def normalize(vol: np.ndarray) -> np.ndarray:
    """按体数据做 1/99 分位截断 + [0,1] 线性归一化（MRI 无统一 HU 标度，需逐例归一）。"""
    lo, hi = np.percentile(vol, (1.0, 99.0))
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def extract_slices(img_path: Path, lab_path: Path, size: int, min_fg: int):
    """把一个 3D 体数据切成 2D 轴向切片，丢掉几乎全空的切片。"""
    vol = np.asarray(nib.load(str(img_path)).dataobj, dtype=np.float32)
    lab = np.asarray(nib.load(str(lab_path)).dataobj, dtype=np.float32)
    if vol.shape != lab.shape:
        raise ValueError(f"shape mismatch {img_path.name}: {vol.shape} vs {lab.shape}")

    vol = normalize(vol)
    imgs, labs = [], []
    for z in range(vol.shape[2]):
        sl, sm = vol[:, :, z], lab[:, :, z]
        # 丢弃背景切片：图像本身近乎全黑的没有重建/分割价值
        if sl.max() < 1e-6:
            continue
        img_r = resize(sl, (size, size), order=1, preserve_range=True, anti_aliasing=True)
        lab_r = resize(sm, (size, size), order=0, preserve_range=True, anti_aliasing=False)
        lab_r = (lab_r > 0.5).astype(np.uint8)
        if lab_r.sum() < min_fg:
            # 分割任务保留有前景的切片；重建任务不看 label，另存全量
            imgs.append((img_r.astype(np.float32), lab_r, False))
        else:
            imgs.append((img_r.astype(np.float32), lab_r, True))
    return imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/Task02_Heart")
    ap.add_argument("--out", default="data/slices")
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--min-fg", type=int, default=64, help="切片被计入分割集所需的最少前景像素")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir, lab_dir = root / "imagesTr", root / "labelsTr"

    cases = sorted(p for p in img_dir.glob("*.nii.gz") if not p.name.startswith("."))
    if not cases:
        raise SystemExit(f"no .nii.gz under {img_dir}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(cases))
    n_test = max(1, int(round(len(cases) * args.test_frac)))
    test_idx = set(order[:n_test].tolist())

    split = {"train": [], "test": []}
    stats = {"train": [0, 0], "test": [0, 0]}  # [切片数, 有前景切片数]

    for i, img_path in enumerate(cases):
        lab_path = lab_dir / img_path.name
        if not lab_path.exists():
            print(f"skip {img_path.name}: no label")
            continue
        tag = "test" if i in test_idx else "train"
        slices = extract_slices(img_path, lab_path, args.size, args.min_fg)
        if not slices:
            continue
        case_id = img_path.name.replace(".nii.gz", "")
        imgs = np.stack([s[0] for s in slices])
        labs = np.stack([s[1] for s in slices])
        fg = np.array([s[2] for s in slices], dtype=bool)
        np.savez_compressed(out / f"{case_id}.npz", img=imgs, lab=labs, fg=fg)
        split[tag].append(case_id)
        stats[tag][0] += len(slices)
        stats[tag][1] += int(fg.sum())
        print(f"{tag:5s} {case_id}: {len(slices)} slices ({int(fg.sum())} with fg)")

    meta = {
        "size": args.size,
        "split": split,
        "n_cases": {k: len(v) for k, v in split.items()},
        "n_slices": {k: v[0] for k, v in stats.items()},
        "n_slices_fg": {k: v[1] for k, v in stats.items()},
        "split_level": "patient",
        "seed": args.seed,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\n" + json.dumps(meta["n_cases"] | {"slices": meta["n_slices"]}, indent=2))


if __name__ == "__main__":
    main()
