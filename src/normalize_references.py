"""Renumber numeric citations by first appearance without touching intervals."""
import argparse
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, required=True)
    args = ap.parse_args()
    text = args.draft.read_text()
    body, tail = text.split("## 参考文献", 1)
    refs, thanks = tail.split("## 致谢", 1)
    definitions = {int(n): line for n, line in
                   re.findall(r"^\[(\d+)\]\s*(.+)$", refs, re.M)}
    mapping = {}

    def replace(match):
        ids = []
        for token in match.group(1).split(","):
            if "-" in token:
                lo, hi = map(int, token.split("-"))
                ids.extend(range(lo, hi + 1))
            else:
                ids.append(int(token))
        if any(i not in definitions for i in ids):
            return match.group(0)
        for i in ids:
            mapping.setdefault(i, len(mapping) + 1)
        return "[" + ",".join(str(mapping[i]) for i in ids) + "]"

    body = re.sub(r"\[([1-9]\d*(?:[,-][1-9]\d*)*)\]", replace, body)
    missing = set(definitions) - set(mapping)
    if missing:
        raise ValueError(f"uncited references: {sorted(missing)}")
    reordered = "\n\n".join(f"[{new}] {definitions[old]}" for old, new in mapping.items())
    args.draft.write_text(body + "## 参考文献\n\n" + reordered + "\n\n---\n\n## 致谢" + thanks)
    print("reference order:", mapping)


if __name__ == "__main__":
    main()
