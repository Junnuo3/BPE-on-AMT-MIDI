#!/usr/bin/env python3
"""Add vocab_proportion to component_analysis entries in existing summary.json files."""

import csv
import json
import re
from pathlib import Path

analysis_root = Path(__file__).parent.parent / "outputs" / "analysis"


def load_vocab_totals(vocab_dir: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    for csv_path in sorted(vocab_dir.glob("component_analysis_size*.csv")):
        m = re.match(r"component_analysis_size(\d+)\.csv$", csv_path.name)
        if not m:
            continue
        size = m.group(1)
        with open(csv_path, newline="") as f:
            totals[size] = sum(int(row["vocab_count"]) for row in csv.DictReader(f))
    return totals


def patch_summary(summary_path: Path, totals: dict[str, int]) -> bool:
    with open(summary_path) as f:
        summary = json.load(f)

    component_analysis = summary.get("component_analysis")
    if not component_analysis:
        return False

    changed = False
    for size, entries in component_analysis.items():
        total_vocab = totals.get(size, 0)
        for entry in entries:
            expected = round(entry["vocab_count"] / total_vocab, 4) if total_vocab else 0.0
            if entry.get("vocab_proportion") != expected:
                entry["vocab_proportion"] = expected
                changed = True

    if changed:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
    return changed


def main() -> None:
    for vocab_dir in sorted(analysis_root.iterdir()):
        if not vocab_dir.is_dir() or not re.match(r"vocab\d+$", vocab_dir.name):
            continue
        summary_path = vocab_dir / "summary.json"
        if not summary_path.exists():
            print(f"{vocab_dir.name}: no summary.json — skip")
            continue
        totals = load_vocab_totals(vocab_dir)
        if not totals:
            print(f"{vocab_dir.name}: no component_analysis CSVs — skip")
            continue
        if patch_summary(summary_path, totals):
            print(f"{vocab_dir.name}: updated")
        else:
            print(f"{vocab_dir.name}: already up to date")


if __name__ == "__main__":
    main()
