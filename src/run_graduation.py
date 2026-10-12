"""Run the graduation experiments sequentially, with persistent logs and resume."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from train_ista import atomic_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slices", default="data/slices")
    ap.add_argument("--device", choices=("mps", "cpu", "auto"), default="mps")
    ap.add_argument("--stages", default="fixed,learnable,phases,report")
    ap.add_argument("--wait-fixed", action="store_true",
                    help="queue after an already running fixed-mask training finishes")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    common = [
        sys.executable, str(repo/"src"/"train_ista.py"),
        "--slices", str(Path(args.slices).resolve()), "--filters", "32",
        "--epochs", "50", "--batch", "8", "--lr", "1e-3", "--mask-lr", "1e-3",
        "--sym-w", "0.01", "--mask-gamma", "10", "--seed", "0", "--mask-seed", "0",
        "--device", args.device, "--resume",
    ]
    stages = args.stages.split(",")
    if set(stages) - {"fixed", "learnable", "phases", "report"}:
        ap.error("unknown stage")
    jobs = []
    for mode in ("fixed", "learnable"):
        if mode in stages:
            jobs.append((mode, common + ["--out", str(args.out),
                        "--mask-mode", mode, "--phases", "9", "--ratios", "0.1,0.2,0.3"]))
    if "phases" in stages:
        for k in (3, 6, 12):
            folder = args.out / "ablation" / f"phases_k{k}"
            jobs.append((f"phases_k{k}", common + [
                "--out", str(folder), "--mask-mode", "fixed", "--phases", str(k),
                "--ratios", "0.2",
            ]))
    if "report" in stages:
        jobs.append(("report", [sys.executable, str(repo/"src"/"report_ista.py"),
                                "--results", str(args.out), "--device", args.device,
                                "--slices", str(Path(args.slices).resolve())]))
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    status = {"started_at": datetime.now(timezone.utc).isoformat(), "jobs": []}
    if args.wait_fixed:
        deadline = time.monotonic() + 36 * 3600
        print("Waiting for the existing fixed-mask run to finish.", flush=True)
        while True:
            path = args.out / "ista_results.json"
            results = json.loads(path.read_text()) if path.exists() else {}
            if all(results.get(tag, {}).get("status") == "complete"
                   for tag in ("cs10", "cs20", "cs30")):
                break
            if time.monotonic() > deadline:
                raise SystemExit("Fixed training did not complete within 36 hours")
            time.sleep(15)
    for name, command in jobs:
        job = {"name": name, "command": command, "status": "running"}
        status["jobs"].append(job)
        atomic_json(args.out / "pipeline.json", status)
        print(f"START {name}", flush=True)
        with (args.out / f"{name}_train.log").open("a") as log:
            process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        job["status"] = "complete" if code == 0 else "failed"
        job["ended_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.out / "pipeline.json", status)
        if code:
            raise SystemExit(f"{name} failed with exit code {code}; state was retained")
    status["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(args.out / "pipeline.json", status)
    print("All requested stages completed.", flush=True)


if __name__ == "__main__":
    main()
