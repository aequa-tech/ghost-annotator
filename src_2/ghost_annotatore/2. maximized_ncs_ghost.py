"""Compute maximized ghost-annotator non conformity score (NCS) per comment.

Input:
- All CSV files in src_2/annotazione_modelli matching results_{MODEL}_{DATASET}.csv

Output:
- One CSV per input in src_2/ghost_annotatore/maximized/{MODEL}_{DATASET}.csv
- Columns: comment_id, ncs

NCS definition (maximized setting):
- Ghost label is the class with minimum probability (argmin over probs).
- Non conformity is the mean Brier component over classes:
    mean_k ( (1 - p_k)^2 if k == y_ghost else p_k^2 )
"""

from __future__ import annotations

import ast
import glob
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / "annotazione_modelli"
OUTPUT_DIR = BASE_DIR / "ghost_annotatore" / "maximized"


def parse_probs(raw: str) -> dict[str, float]:
    """Parse probs serialized as JSON-like string into {label: prob} dict."""
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}

    text = str(raw).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)

    return {str(k): float(v) for k, v in data.items()}


def ghost_ncs_maximized(probs: dict[str, float]) -> float:
    """Compute non conformity score with ghost label = argmin(probs)."""
    if not probs:
        raise ValueError("Empty probs dictionary")

    #qui si usa min invece che max (1. minimized_ncs_ghost.py)
    ghost_label = min(probs.items(), key=lambda item: item[1])[0]
    components = []

    for label, prob in probs.items():
        if label == ghost_label:
            components.append((1.0 - prob) ** 2)
        else:
            components.append(prob**2)

    return float(sum(components) / len(components))


def process_one_file(input_path: Path) -> None:
    df = pd.read_csv(input_path)

    required_cols = {"comment_id", "probs"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {input_path.name}")

    df = df[["comment_id", "probs"]].copy()
    df["ncs"] = df["probs"].apply(lambda x: ghost_ncs_maximized(parse_probs(x)))

    # Ensure one row per comment_id.
    out_df = (
        df.groupby("comment_id", as_index=False)["ncs"]
        .mean()
        .sort_values("comment_id")
        .reset_index(drop=True)
    )

    # results_{MODEL}_{DATASET}.csv -> {MODEL}_{DATASET}.csv
    out_name = input_path.name.replace("results_", "", 1)
    out_path = OUTPUT_DIR / out_name
    out_df.to_csv(out_path, index=False)

    print(f"Saved {out_name}: {len(out_df)} comments")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(glob.glob(str(INPUT_DIR / "results_*.csv")))

    if not input_files:
        print(f"No input files found in {INPUT_DIR}")
        return

    for file_path in input_files:
        process_one_file(Path(file_path))


if __name__ == "__main__":
    main()
