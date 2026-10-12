"""Idempotently update numeric tables only, preserving reviewed thesis prose."""
import argparse
import json
import re
from pathlib import Path

from report_ista import markdown_table, tables

CAPTIONS = {
    "psnr_table": "表5-4  全图口径PSNR对比",
    "ssim_table": "表5-5  全图口径SSIM对比",
    "roi_table": "表5-6  ROI体内口径PSNR对比",
    "phase_table": "表5-7  独立训练的Phase数量消融",
    "dc_table": "表5-8  同一checkpoint的末尾DC推理消融",
}


def measured_tables(report):
    if report.get("status") != "complete" or report.get("missing"):
        raise ValueError("thesis_results.json is incomplete")
    result = tables(report)
    dc = report["hard_dc"]
    result["dc_table"] = markdown_table(
        ["推理配置（同一checkpoint）", "PSNR（dB）", "SSIM"],
        [[label, f"{dc[key]['psnr']:.2f}", f"{dc[key]['ssim']:.4f}"]
         for label, key in (("关闭末尾硬DC", "without_hard_dc"),
                            ("开启末尾硬DC", "with_hard_dc"))])
    return result


def replace_tables(text, replacements):
    for key, caption in CAPTIONS.items():
        start = text.index(caption)
        match = re.search(r"^\|[^\n]*(?:\n\|[^\n]*)*", text[start:], re.M)
        if match is None:
            raise ValueError(f"missing table after {caption}")
        i, j = start + match.start(), start + match.end()
        text = text[:i] + replacements[key] + text[j:]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--draft", type=Path, required=True)
    args = ap.parse_args()
    text = replace_tables(args.draft.read_text(),
                          measured_tables(json.loads(args.report.read_text())))
    args.draft.write_text(text)
    print(f"updated five measured tables: {args.draft}")


if __name__ == "__main__":
    main()
