"""Run compact ISTA-Net+ ablations and write one machine-readable result file.

Examples
--------
Phase-count ablation:
    python src/run_ista_ablation.py --kind phases --ratios 0.2 --phases 3,6,9,12

Hard data-consistency ablation:
    python src/run_ista_ablation.py --kind hard-dc --ratios 0.2 --phases 9,9

The script deliberately delegates each experiment to ``train_ista.py`` so the
same corrected complex-domain DC implementation and metric aggregation are used
for the main experiment and the ablations.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("phases", "hard-dc"), required=True)
    ap.add_argument("--slices", default="data/slices")
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--ratios", default="0.2")
    ap.add_argument("--phases", default="3,6,9,12")
    ap.add_argument("--filters", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sym-w", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    ratios = args.ratios
    if args.kind == "phases":
        settings = [
            (int(k), True, f"phases_k{k}")
            for k in args.phases.split(",")
        ]
    else:
        settings = [
            (int(args.phases.split(",")[0]), True, "hard_dc"),
            (int(args.phases.split(",")[-1]), False, "no_hard_dc"),
        ]

    summary = {}
    for phases, hard_dc, name in settings:
        exp_out = out_dir / name
        cmd = [
            python,
            "src/train_ista.py",
            "--slices",
            args.slices,
            "--out",
            str(exp_out),
            "--ratios",
            ratios,
            "--mask-mode",
            "fixed",
            "--phases",
            str(phases),
            "--filters",
            str(args.filters),
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
            "--lr",
            str(args.lr),
            "--sym-w",
            str(args.sym_w),
            "--seed",
            str(args.seed),
            "--mask-seed",
            str(args.mask_seed),
            "--device",
            args.device,
        ]
        if not hard_dc:
            cmd.append("--no-final-hard-dc")
        if args.resume:
            cmd.append("--resume")
        _run(cmd)

        result_path = exp_out / "ista_results.json"
        result = json.loads(result_path.read_text())
        summary[name] = {
            "kind": args.kind,
            "phases": phases,
            "final_hard_dc": hard_dc,
            "results": result,
        }
        (out_dir / f"{args.kind}_results.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
