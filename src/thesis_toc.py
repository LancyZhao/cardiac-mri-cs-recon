"""Map headings to printed body page numbers from an actual PDF rendering."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def normalized(text):
    return re.sub(r"\s+", "", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    text = args.draft.read_text()
    text = text[text.index("## 第一章"):]
    headings = re.findall(r"^#{2,4}\s+(.+)$", text, re.M)
    pages = subprocess.check_output(["pdftotext", "-layout", str(args.pdf), "-"]).decode().split("\f")
    page_map = {}
    physical = 0
    # Ignore the cached table of contents by finding the body header first.
    for i, page in enumerate(pages):
        if "天津大学" in page and re.search(r"2026\s*届", page) and "第一章" in page:
            physical = i
            break
    body_start = physical
    offset = 0
    for heading in headings:
        key = normalized(heading)
        while physical < len(pages):
            page = normalized(pages[physical])
            pos = page.find(key, offset)
            if pos >= 0:
                page_map[key] = physical - body_start + 1
                offset = pos + len(key)
                break
            physical += 1
            offset = 0
        else:
            raise ValueError(f"heading not found in rendered PDF: {heading}")
    args.output.write_text(json.dumps(page_map, indent=2, ensure_ascii=False))
    print(f"mapped {len(page_map)} headings, body starts at PDF page {body_start + 1}")


if __name__ == "__main__":
    main()
