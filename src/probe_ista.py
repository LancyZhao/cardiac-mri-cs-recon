"""Numerical evidence on a named, reproducible subset, separate from test metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ista_net import (
    ISTANetPlus, LearnableMask, cartesian_dc_gradient,
    cartesian_hard_dc, dc_gradient, hard_dc,
    simulate_cartesian_measurement, simulate_measurement,
)
from sampling import cartesian_mask, psnr
from train_ista import _device, atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", type=Path, default=Path("data/slices"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    args = ap.parse_args()
    device = _device(args.device)
    meta = json.loads((args.slices / "meta.json").read_text())
    cid = meta["split"]["test"][0]
    with np.load(args.slices / f"{cid}.npz") as d:
        x = torch.from_numpy(d["img"][:16].copy())[:, None].to(device)
    mask = torch.from_numpy(cartesian_mask(meta["size"], 0.1)).to(device)
    with torch.no_grad():
        z2, k2 = simulate_measurement(x, mask)
        z1, k1 = simulate_cartesian_measurement(x, mask)
        gd = z1.clone()
        for _ in range(9):
            gd = gd - 0.5 * cartesian_dc_gradient(gd, k1, mask)
        dc = cartesian_hard_dc(z1, k1, mask)
        pred = {"zf": z1, "hard_dc": dc, "gd_k9_rho0.5": gd}
        metrics = {name: float(np.mean([
            psnr(g[0], p[0]) for g, p in
            zip(x.cpu().numpy(), output.cpu().numpy())]))
            for name, output in pred.items()}
        gradient2 = dc_gradient(z2, k2, mask)
        error = {
            "zf": (z1-z2).abs().max().item(),
            "gradient": (cartesian_dc_gradient(z2, k1, mask)-gradient2).abs().max().item(),
            "hard_dc": (cartesian_hard_dc(z2,k1,mask)-hard_dc(z2,k2,mask)).abs().max().item(),
            "batch_single": (gradient2[:1]-dc_gradient(z2[:1],k2[:1],mask)).abs().max().item(),
        }
    torch.manual_seed(0)
    lm = LearnableMask(meta["size"], 0.1, mask).to(device)
    net = ISTANetPlus(1, 4).to(device)
    measured_mask = lm()
    z, k = simulate_cartesian_measurement(x[:2], measured_mask)
    p, sym = net(z, measured_mask, k, fast_cartesian=True)
    ((p-x[:2]).abs().mean() + 0.01*sym + 10*lm.rate_penalty()).backward()
    gradients = lm.logits.grad.detach()
    evidence = {
        "device": str(device), "case": cid, "slice_indices": list(range(16)),
        "nominal_ratio": 0.1, "seed": 0, "psnr_subset": metrics,
        "operator_max_absolute_errors": error,
        "mps_joint_gradient_finite": bool(torch.isfinite(gradients).all().item()),
        "mask_gradient_absolute_sum": gradients.abs().sum().item(),
        "acs_gradient_absolute_sum": gradients[lm.acs_lines].abs().sum().item(),
        "mask": lm.summary(),
        "note": "Numerical sanity check on first 16 slices; NOT full-test trained metrics.",
    }
    assert max(error.values()) < 1e-5, error
    assert evidence["mps_joint_gradient_finite"]
    assert evidence["mask_gradient_absolute_sum"] > 0
    assert evidence["acs_gradient_absolute_sum"] == 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, evidence)
    print(json.dumps({k:v for k,v in evidence.items() if k != "mask"}, indent=2))


if __name__ == "__main__":
    main()
