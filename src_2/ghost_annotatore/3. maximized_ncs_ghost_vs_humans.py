"""Compute maximized ghost NCS per comment against human annotations.

For each comment c, let P_c(y) be the probability assigned by the model to label y.
Let H_c be the set of labels chosen by the human annotators for that comment.

Ghost-label selection rule:
- If there is at least one label never chosen by humans, select
            y_ghost = argmin_{y not in H_c} P_c(y)
    and set empty_label = true.
- If all labels were chosen by humans, select
            y_ghost = argmin_y P_c(y)
    and set empty_label = false.

The non conformity score (NCS) is computed with the same Brier-style definition
used in the main project, treating y_ghost as the target label:

        NCS(c) = (1 / K) * sum_{y in Y} s(y)

where K is the number of labels, Y is the label set, and

        s(y) = (1 - P_c(y))^2   if y = y_ghost
        s(y) = (P_c(y))^2       otherwise

Equivalently:

        NCS(c) = (1 / K) * [ (1 - P_c(y_ghost))^2
                                                 + sum_{y != y_ghost} (P_c(y))^2 ]

Input:
- src_2/annotazione_modelli/results_{MODEL}_{DATASET}.csv (16 files)
- src_2/data/*.csv (4 human datasets)

Output:
- src_2/ghost_annotatore/maximized_human/{MODEL}_{DATASET}.csv
- Columns: comment_id, ncs, empty_label
"""

from __future__ import annotations

import ast
import glob
import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_MODEL_DIR = BASE_DIR / "annotazione_modelli"
INPUT_HUMAN_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "ghost_annotatore" / "maximized_human"

HUMAN_DATASET_FILES = {
    "attitudes": "attitudes - hate speech.csv",
    "cade": "cade - acceptability.csv",
    "davani": "davani - offensiveness.csv",
    "measuring": "measuring - violence.csv",
}


def normalize_label(value) -> str:
    """Normalize labels to comparable string form, e.g. 1.0 -> '1'."""
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text == "":
        return ""

    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return text


def parse_probs(raw: str) -> dict[str, float]:
    """Parse probs serialized as JSON-like string into {label: prob} dict."""
    if isinstance(raw, dict):
        return {normalize_label(k): float(v) for k, v in raw.items()}

    text = str(raw).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)

    return {normalize_label(k): float(v) for k, v in data.items()}


def compute_ncs_for_label(probs: dict[str, float], chosen_label: str) -> float:
    """Brier-style non conformity score for a chosen label.

    Formula:
        NCS = (1 / K) * [ (1 - p_chosen)^2 + sum_{y != chosen} p_y^2 ]
    where K is the number of labels.
    """
    components = []
    for label, prob in probs.items():
        if label == chosen_label:
            components.append((1.0 - prob) ** 2)
        else:
            components.append(prob**2)
    return float(sum(components) / len(components))


def select_ghost_label(probs: dict[str, float], human_labels: set[str]) -> tuple[str, bool]:
    """Select ghost label using the 'empty label first' rule."""
    available_labels = list(probs.keys())
    empty_candidates = [lab for lab in available_labels if lab not in human_labels]

    if empty_candidates:
        chosen = min(empty_candidates, key=lambda lab: probs[lab])
        return chosen, True

    chosen = min(available_labels, key=lambda lab: probs[lab])
    return chosen, False


def load_human_label_sets(dataset_key: str) -> dict[str, set[str]]:
    """Return {comment_id: set(labels_used_by_humans)} for one dataset."""
    human_file = INPUT_HUMAN_DIR / HUMAN_DATASET_FILES[dataset_key]
    df = pd.read_csv(human_file, usecols=["comment_id", "label"])

    df = df.dropna(subset=["comment_id", "label"]).copy()
    df["comment_id"] = df["comment_id"].astype(str)
    df["label"] = df["label"].apply(normalize_label)

    label_map = (
        df.groupby("comment_id")["label"]
        .apply(lambda s: set(x for x in s.tolist() if x != ""))
        .to_dict()
    )
    return label_map


def process_one_model_file(input_path: Path, human_label_map: dict[str, set[str]]) -> None:
    df = pd.read_csv(input_path, usecols=["comment_id", "probs"])
    df = df.dropna(subset=["comment_id", "probs"]).copy()
    df["comment_id"] = df["comment_id"].astype(str)

    rows = []
    for _, row in df.iterrows():
        comment_id = row["comment_id"]
        probs = parse_probs(row["probs"])
        human_labels = human_label_map.get(comment_id, set())

        ghost_label, empty_label = select_ghost_label(probs, human_labels)
        ncs = compute_ncs_for_label(probs, ghost_label)

        rows.append(
            {
                "comment_id": comment_id,
                "ncs": ncs,
                "empty_label": "true" if empty_label else "false",
            }
        )

    out_df = pd.DataFrame(rows)

    # One row per comment_id (stable in case of repeated rows in model files).
    out_df = (
        out_df.groupby("comment_id", as_index=False)
        .agg(
            ncs=("ncs", "mean"),
            empty_label=("empty_label", "first"),
        )
        .sort_values("comment_id")
        .reset_index(drop=True)
    )

    out_name = input_path.name.replace("results_", "", 1)
    out_path = OUTPUT_DIR / out_name
    out_df.to_csv(out_path, index=False)

    print(f"Saved {out_name}: {len(out_df)} comments")


def parse_dataset_key_from_filename(filename: str) -> str:
    """results_{MODEL}_{DATASET}.csv -> DATASET"""
    core = filename.replace("results_", "", 1).replace(".csv", "")
    _, dataset_key = core.rsplit("_", 1)
    return dataset_key


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(glob.glob(str(INPUT_MODEL_DIR / "results_*.csv")))
    if not input_files:
        print(f"No input files found in {INPUT_MODEL_DIR}")
        return

    # Load human labels once per dataset.
    human_maps = {key: load_human_label_sets(key) for key in HUMAN_DATASET_FILES}

    for file_path in input_files:
        path = Path(file_path)
        dataset_key = parse_dataset_key_from_filename(path.name)
        if dataset_key not in human_maps:
            raise ValueError(f"Unknown dataset key '{dataset_key}' in {path.name}")

        process_one_model_file(path, human_maps[dataset_key])


if __name__ == "__main__":
    main()
