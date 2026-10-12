"""Record and preserve the one-off 2-D to 1-D Cartesian checkpoint migration."""
import argparse
import shutil
from pathlib import Path

import torch

from train_ista import atomic_checkpoint, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--slices", type=Path, default=Path("data/slices"))
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    backup = args.checkpoint.with_name(args.checkpoint.stem + "_before_migration.pt")
    if not backup.exists():
        shutil.copy2(args.checkpoint, backup)
    previous = state.pop("provenance_migration", None)
    migrations = state.get("migrations", [])
    if previous:
        migrations.append({**previous, "after_epoch": state["epoch"] + 1})
    migrations.append({
        "after_epoch": state["epoch"] + 1,
        "from": state["provenance"],
        "to": provenance(args.slices),
        "reason": "Carry migration metadata into future states, model and result. "
                  "Cartesian 1-D forward and STE gradients verified against 2-D.",
    })
    state["provenance"] = migrations[-1]["to"]
    state["migrations"] = migrations
    atomic_checkpoint(args.checkpoint, state)
    print(f"Preserved epoch {state['epoch'] + 1}; backup: {backup}")


if __name__ == "__main__":
    main()
