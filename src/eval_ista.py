"""Reevaluate every saved ISTA checkpoint and compare all reported metrics."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from torch.utils.data import DataLoader

from report_ista import collect, load_model
from train_ista import ReconSet, _device, atomic_json, evaluate, provenance


def differences(expected, actual, prefix=""):
    for key, value in expected.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from differences(value, actual[key], path)
        elif isinstance(value, (float, int)):
            yield path, abs(value - actual[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=Path("results"))
    ap.add_argument("--slices", type=Path, default=Path("data/slices"))
    ap.add_argument("--device", choices=("cpu", "mps", "auto"), default="auto")
    ap.add_argument("--tolerance", type=float, default=1e-4)
    args = ap.parse_args()
    device = _device(args.device)
    data = collect(args.results, args.baseline)
    fingerprint = provenance(args.slices)
    audit = {"evaluated_at": datetime.now(timezone.utc).isoformat(),
             "device": str(device), "tolerance": args.tolerance, "runs": {}}
    jobs = [(f"{mode}/{tag}", e, args.results)
            for mode in ("fixed", "learnable") for tag, e in data[mode].items()]
    jobs += [(f"phases/{k}", e, args.results / "ablation" / f"phases_k{k}")
             for k, e in data["phases"].items() if k != "9"]
    for name, entry, folder in jobs:
        if entry["provenance"]["sha256"] != fingerprint["sha256"]:
            raise ValueError(f"source or data fingerprint mismatch: {name}")
        model, mask = load_model(entry, folder, device)
        # Match the original cache-vs-dynamic measurement path, including rounding.
        ds = ReconSet(args.slices, entry["config"]["split"]["test"],
                      mask=mask.cpu().numpy() if entry["mask_mode"] == "fixed" else None)
        dl = DataLoader(ds, batch_size=entry["batch"], shuffle=False)
        actual = evaluate(model, dl, mask, device, ds.case_of)
        delta = dict(differences(entry["metrics"], actual))
        audit["runs"][name] = {"metrics": actual, "maximum_difference": max(delta.values()),
                               "passed": max(delta.values()) <= args.tolerance}
        print(name, audit["runs"][name]["maximum_difference"], flush=True)
        atomic_json(args.results / "reevaluation.json", audit)
    audit["passed"] = all(v["passed"] for v in audit["runs"].values())
    atomic_json(args.results / "reevaluation.json", audit)
    if not audit["passed"]:
        raise SystemExit("checkpoint reevaluation differs from the saved report")


if __name__ == "__main__":
    main()
