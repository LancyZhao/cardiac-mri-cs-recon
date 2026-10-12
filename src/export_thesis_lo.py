"""Build Word and PDF with embedded fonts and render-verified cached TOC fields."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from build_thesis_docx import build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--soffice", type=Path, default=Path.home() /
                    "Applications/LibreOffice.app/Contents/MacOS/soffice")
    args = ap.parse_args()
    if not args.soffice.is_file():
        raise SystemExit("LibreOffice executable not found; supply --soffice")
    source_dir = Path(__file__).resolve().parent
    pdf = args.output.with_suffix(".pdf")
    toc = args.output.with_suffix(".toc.json")
    page_map = json.loads(toc.read_text()) if toc.exists() else {}
    env = dict(os.environ)
    env["FONTCONFIG_FILE"] = str(source_dir / "thesis_fonts.conf")
    for iteration in range(4):
        build(args.source, args.output, template=args.template, toc_map=page_map)
        with tempfile.TemporaryDirectory(prefix="thesis-render-") as profile:
            subprocess.run([
                str(args.soffice), "-env:UserInstallation=" + Path(profile).as_uri(),
                "--headless", "--nologo", "--nodefault", "--norestore",
                "--convert-to", "pdf:writer_pdf_Export", "--outdir",
                str(pdf.parent), str(args.output),
            ], env=env, check=True, timeout=180)
        subprocess.run([
            sys.executable, str(source_dir / "thesis_toc.py"),
            "--draft", str(args.source), "--pdf", str(pdf), "--output", str(toc),
        ], check=True, timeout=30)
        updated = json.loads(toc.read_text())
        if updated == page_map:
            print(f"TOC verified after {iteration + 1} renders: {pdf}", flush=True)
            return
        page_map = updated
    raise SystemExit("pagination did not stabilize; inspect the rendered PDF")


if __name__ == "__main__":
    main()
