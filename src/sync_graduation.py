"""Copy verified deliverables without deleting historical artifacts."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and source.read_bytes() == target.read_bytes():
        return
    shutil.copy2(source, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-root", type=Path, required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    results = args.paper_root / "结果_修复版"
    report = json.loads((results / "thesis_results.json").read_text())
    if report["status"] != "complete":
        raise SystemExit("cannot deliver incomplete experiments")
    package = args.paper_root / "改动代码以及结果"
    for name in ("ista_net.py", "train_ista.py"):
        existing = package / name
        backup = package / "历史代码_同步前" / name
        if existing.exists() and not backup.exists():
            copy(existing, backup)
    for source in sorted((repo / "src").iterdir()):
        if source.is_file():
            copy(source, package / "src" / source.name)
    for name in ("ista_net.py", "train_ista.py"):
        copy(repo / "src" / name, package / name)
    for name in ("requirements.txt", "requirements-docs.txt", "README.md", "repro.sh"):
        copy(repo / name, package / name)
    for source in (repo / "tests").glob("test_*.py"):
        copy(source, package / "tests" / source.name)
    public = repo / "results" / "graduation"
    files = [p for p in results.glob("*.json")]
    files += [results / "REPORT.md"]
    files += list((results / "figures").glob("*.png")) + list((results / "figures").glob("*.pdf"))
    files += [p for p in results.glob("*_train.log")]
    files += [results / e["checkpoint"] for group in ("fixed", "learnable")
              for e in report[group].values()]
    for k in (3, 6, 12):
        folder = results / "ablation" / f"phases_k{k}"
        files += list(folder.glob("*.json")) + [folder / "ista_cs20.pt"]
    manifest = {}
    for source in sorted(set(files)):
        relative = source.relative_to(results)
        copy(source, public / relative)
        manifest[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    # The local handover includes optimizer states too; public checkpoints are inference-only.
    shutil.copytree(results, package / "results", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("pdf_profile", "pdf", "*.docx", "*.before-rebuild.docx"))
    for source in (repo / "results").glob("recon*"):
        if source.is_file():
            copy(source, package / "baseline" / source.name)
    (public / "SHA256SUMS.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2))
    local_manifest = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS.json" and "__pycache__" not in p.parts
    }
    (package / "SHA256SUMS.json").write_text(
        json.dumps(local_manifest, ensure_ascii=False, indent=2))
    print(f"synced {len(manifest)} public experiment artifacts and local source/state package")


if __name__ == "__main__":
    main()
