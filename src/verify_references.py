"""Fetch publisher-deposited metadata for manual bibliography verification."""
import argparse
import json
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError


TITLES = [
    "ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing",
    "Sparse MRI: The application of compressed sensing for rapid MR imaging",
    "Robust uncertainty principles: Exact signal reconstruction from highly incomplete frequency information",
    "10.1109/TIT.2006.871582",
    "U-Net: Convolutional Networks for Biomedical Image Segmentation",
    "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems",
    "End-to-End Variational Networks for Accelerated MRI Reconstruction",
    "Image quality assessment: from error visibility to structural similarity",
    "Learning the sampling pattern for MRI",
    "MoDL: Model-Based Deep Learning Architecture for Inverse Problems",
    "Deep-Learning-Based Optimization of the Under-Sampling Pattern in MRI",
    "SwinMR: Image reconstruction using Swin Transformer for fast MRI",
    "DAGAN: Deep De-Aliasing Generative Adversarial Networks for Fast Compressed Sensing MRI Reconstruction",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    records = json.loads(args.output.read_text())["records"] if args.output.exists() else []
    for title in TITLES:
        if any(r["query"] == title for r in records):
            continue
        url = "https://api.crossref.org/works?" + urlencode({
            "query.title": title, "rows": 3,
            "select": "title,author,DOI,container-title,volume,issue,page,published,abstract,link",
        })
        if title.startswith("10."):
            url = "https://api.crossref.org/works/" + title
        req = Request(url, headers={"User-Agent": "MRI-Thesis-Bibliography-Audit/1.0"})
        for attempt in range(4):
            try:
                with urlopen(req, timeout=60) as r:
                    result = json.load(r)["message"]
                    hits = [result] if title.startswith("10.") else result["items"]
                break
            except HTTPError as error:
                if error.code not in (429, 502, 503) or attempt == 3:
                    raise
                time.sleep(30 * (attempt + 1))
        records.append({"query": title, "candidates": hits})
        print(title, flush=True)
        for hit in hits[:2]:
            authors = ", ".join(x.get("family", "") for x in hit.get("author", [])[:3])
            print(hit.get("title"), authors, hit.get("DOI"), hit.get("volume"),
                  hit.get("page"), hit.get("published"), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"checked_on": str(date.today()),
                                         "source": "Crossref publisher deposits",
                                         "records": records}, indent=2, ensure_ascii=False))
        time.sleep(5)


if __name__ == "__main__":
    main()
