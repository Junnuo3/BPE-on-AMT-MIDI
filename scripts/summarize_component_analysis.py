#!/usr/bin/env python3
"""Aggregate component_analysis_size*.csv files across all vocab sizes into one summary CSV."""

import csv
import re
from pathlib import Path

analysis_dir = Path(__file__).parent.parent / "outputs" / "analysis"
output_path = analysis_dir / "component_analysis_summary.csv"

rows = []
for vocab_dir in sorted(analysis_dir.iterdir()):
    if not vocab_dir.is_dir():
        continue
    m = re.match(r"vocab(\d+)$", vocab_dir.name)
    if not m:
        continue
    vocab_size = int(m.group(1))

    for csv_path in sorted(vocab_dir.glob("component_analysis_size*.csv")):
        sm = re.match(r"component_analysis_size(\d+)\.csv$", csv_path.name)
        if not sm:
            continue
        token_size = int(sm.group(1))

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({
                    "vocab_size": vocab_size,
                    "token_size": token_size,
                    "categories": row["categories"],
                    "vocab_count": int(row["vocab_count"]),
                    "corpus_count": int(row["corpus_count"]),
                })

rows.sort(key=lambda r: (r["vocab_size"], r["token_size"], -r["corpus_count"]))

with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["vocab_size", "token_size", "categories", "vocab_count", "corpus_count"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {output_path}")
