"""
Reads all output_def/results_{MODEL}_{DATASET}.csv files,
keeps only comment_id, label, probs columns (no duplicates),
and saves them into this folder (annotazione_modelli/).
"""

import glob
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_DEF_DIR = os.path.join(REPO_ROOT, "output_def")

result_files = sorted(glob.glob(os.path.join(OUTPUT_DEF_DIR, "results_*.csv")))

if not result_files:
    print(f"No results_*.csv files found in {OUTPUT_DEF_DIR}")
else:
    for filepath in result_files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath, usecols=["comment_id", "label", "probs"])
        df = df.drop_duplicates()
        dest_path = os.path.join(SCRIPT_DIR, filename)
        df.to_csv(dest_path, index=False)
        print(f"Saved {filename}: {len(df)} rows")
