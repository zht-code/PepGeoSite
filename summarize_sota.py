#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine PepGeoSite benchmark reports.")
    parser.add_argument("--result-dir", default="results/sota_retest")
    parser.add_argument("--benchmarks", nargs="+", default=["TS251", "TS639", "TS125"])
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    rows = []
    for benchmark in args.benchmarks:
        with (result_dir / f"{benchmark}.json").open(encoding="utf-8") as handle:
            report = json.load(handle)
        row = {
            "benchmark": benchmark,
            "actual_complexes": report["actual_complexes"],
            "paper_nominal_complexes": report["nominal_complexes_in_paper"],
            "checkpoint_epoch": report["checkpoint_epoch"],
            "fixed_validation_threshold": report["threshold"],
        }
        reference = report["paper_table1_reference"]
        pooled = report["pooled"]
        for metric in ["sensitivity", "specificity", "precision", "mcc", "auroc"]:
            row[f"retest_{metric}"] = pooled[metric]
            row[f"paper_{metric}"] = reference[metric]
            row[f"delta_{metric}"] = pooled[metric] - reference[metric]
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(result_dir / "sota_summary.csv", index=False)
    with (result_dir / "sota_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
