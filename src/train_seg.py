"""实验2：左心房分割。U-Net baseline vs 调优版（损失函数 + 增强 + LR 策略 A/B）。

对应中期表里"调整模型结构、优化器、损失函数等参数"——这里把它做成可复现的对照实验。
指标用 Dice：按病例聚合交集/并集再对 6 例测试病例取平均（比逐切片平均更稳，
不会被大量小前景切片带偏）。
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
from unet import UNet, count_params  # noqa: E402


class SegSet(Dataset):
    def __init__(self, slice_dir: Path, case_ids, augment=False, fg_only=False):
        self.augment = augment
        imgs, labs, cases = [], [], []
        for cid in case_ids:
            d = np.load(slice_dir / f"{cid}.npz")
            im, lb, fg = d["img"], d["lab"], d["fg"]
            if fg_only:
                im, lb = im[fg], lb[fg]
            imgs.append(im)
            labs.append(lb)
            cases += [cid] * len(im)
        self.imgs = np.concatenate(imgs, 0)
        self.labs = np.concatenate(labs, 0)
        self.cases = cases

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im, lb = self.imgs[i], self.labs[i]
        if self.augment:
            if np.random.rand() < 0.5:
                im, lb = np.fliplr(im).copy(), np.fliplr(lb).copy()
            if np.random.rand() < 0.5:
                im, lb = np.flipud(im).copy(), np.flipud(lb).copy()
            k = np.random.randint(4)
            if k:
                im, lb = np.rot90(im, k).copy(), np.rot90(lb, k).copy()
        return torch.from_numpy(im)[None], torch.from_numpy(lb.astype(np.float32))[None]


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return (1 - num / den).mean()


@torch.no_grad()
def evaluate(model, slice_dir, case_ids, device, thr=0.5):
    """按病例聚合 Dice：整例的交集/体积和，再对病例取平均。"""
    model.eval()
    per_case, slice_dices = {}, []
    for cid in case_ids:
        d = np.load(slice_dir / f"{cid}.npz")
        im, lb = d["img"], d["lab"].astype(np.float32)
        inter = psum = tsum = 0.0
        for s in range(0, len(im), 16):
            x = torch.from_numpy(im[s:s + 16])[:, None].to(device)
            p = (torch.sigmoid(model(x)).cpu().numpy()[:, 0] > thr).astype(np.float32)
            t = lb[s:s + 16]
            inter += float((p * t).sum()); psum += float(p.sum()); tsum += float(t.sum())
            for j in range(len(t)):
                if t[j].sum() > 0:
                    slice_dices.append((2 * (p[j] * t[j]).sum() + 1e-6)
                                       / (p[j].sum() + t[j].sum() + 1e-6))
        per_case[cid] = (2 * inter + 1e-6) / (psum + tsum + 1e-6)
    return {
        "dice_case_mean": float(np.mean(list(per_case.values()))),
        "dice_case_std": float(np.std(list(per_case.values()))),
        "dice_slice_mean_fg": float(np.mean(slice_dices)),
        "per_case": {k: float(v) for k, v in per_case.items()},
    }


def run_cfg(name, cfg, slice_dir, meta, args, device):
    print(f"\n--- {name}: {cfg} ---", flush=True)
    tr = SegSet(slice_dir, meta["split"]["train"], augment=cfg["augment"])
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)

    torch.manual_seed(args.seed)
    model = UNet(1, 1, base=args.base, residual=False).to(device)
    if cfg["optim"] == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
             if cfg["cosine"] else None)
    bce = nn.BCEWithLogitsLoss()

    hist = []
    for ep in range(args.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, y)
            if cfg["loss"] == "bce_dice":
                loss = loss + dice_loss(logits, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x); n += len(x)
        if sched:
            sched.step()
        hist.append(tot / max(n, 1))
        print(f"  {name} ep{ep+1:02d}/{args.epochs} loss={hist[-1]:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    res = evaluate(model, slice_dir, meta["split"]["test"], device)
    torch.save(model.state_dict(), Path(args.out) / f"seg_{name}.pt")
    print(f"  {name} -> Dice(case)={res['dice_case_mean']:.4f} "
          f"Dice(fg slice)={res['dice_slice_mean_fg']:.4f}", flush=True)
    return {"config": cfg, "params_M": count_params(model) / 1e6,
            "epochs": args.epochs, "train_slices": len(tr),
            "loss_hist": hist, "metrics": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default="data/slices")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    slice_dir = Path(args.slices)
    meta = json.loads((slice_dir / "meta.json").read_text())
    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}", flush=True)

    configs = {
        # baseline：最朴素的配置——BCE、Adam、恒定 LR、无增强
        "baseline": {"loss": "bce", "optim": "adam", "lr": 1e-3,
                     "cosine": False, "augment": False},
        # 只换损失：BCE + Dice（前景占比低，纯 BCE 被背景主导）
        "loss_dice": {"loss": "bce_dice", "optim": "adam", "lr": 1e-3,
                      "cosine": False, "augment": False},
        # 换优化器：SGD+momentum，看是否真的不如 Adam
        "optim_sgd": {"loss": "bce_dice", "optim": "sgd", "lr": 1e-2,
                      "cosine": False, "augment": False},
        # 调优版：BCE+Dice、Adam、cosine 退火、翻转/旋转增强
        "tuned": {"loss": "bce_dice", "optim": "adam", "lr": 1e-3,
                  "cosine": True, "augment": True},
        # 去混淆版：把损失固定成 BCE，单独看 cosine + 增强值多少 Dice
        # （loss_dice 实测拖低 Dice，若 tuned 直接对比 baseline 会分不清是谁的作用）
        "tuned_bce": {"loss": "bce", "optim": "adam", "lr": 1e-3,
                      "cosine": True, "augment": True},
    }
    if args.only:
        want = set(args.only.split(","))
        configs = {k: v for k, v in configs.items() if k in want}

    out = {}
    res_path = Path(args.out, "seg_results.json")
    if res_path.exists():
        # 合并而非覆盖：--only 单跑某个配置时不该抹掉已有结果
        out = json.loads(res_path.read_text())
    for name, cfg in configs.items():
        out[name] = run_cfg(name, cfg, slice_dir, meta, args, device)
        res_path.write_text(json.dumps(out, indent=2))

    print("\n=== 分割结果（测试集 %d 例，按病例划分）===" % meta["n_cases"]["test"])
    print(f"{'配置':<12} {'Dice(病例)':>12} {'Dice(前景切片)':>16} {'损失':>10} {'优化器':>8}")
    for k, v in out.items():
        m = v["metrics"]
        print(f"{k:<12} {m['dice_case_mean']:>12.4f} {m['dice_slice_mean_fg']:>16.4f} "
              f"{v['config']['loss']:>10} {v['config']['optim']:>8}")


if __name__ == "__main__":
    main()
