"""Figura 2 Bottom (versione src_2): correlazione tra Ghost NCS e annotator isolation.

Questa figura e' analoga alla Figura 2 Top, ma usa l'isolamento degli annotatori
invece della fraction of agreement diretta.

Definizioni:
1) Label del modello per commento:
    y_model(c) = argmax_y P_c(y)
2) Filtro ghost (commenti mantenuti):
    y_model(c) not in H(c)
3) Ghost label e NCS (stessa regola del file maximized_ncs_ghost_vs_humans.py):
    y_ghost(c) = argmin_{y not in H(c)} P_c(y) se possibile, altrimenti argmin_y P_c(y)
    NCS_ghost(c) = (1 / K) * [ (1 - P_c(y_ghost(c)))^2 + sum_{y != y_ghost(c)} P_c(y)^2 ]
4) Fraction of agreement del commento:
    FA(c) = 100 * max_y n_c(y) / N_c
5) Annotator isolation del commento:
    ISO(c) = 100 - FA(c)
    (se FA e' in scala [0,1], allora ISO(c) = 1 - FA(c))
6) Correlazione per ogni modello-dataset:
    r = corr_pearson( NCS_ghost(c), ISO(c) )

La heatmap mostra r (e p-value) per tutte le combinazioni modello-dataset.
"""

import ast
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTAZIONI_DIR = os.path.join(SCRIPT_DIR, 'annotazione_modelli')
HUMAN_DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
IMG_DIR = os.path.join(SCRIPT_DIR, 'img')

HUMAN_DATASET_FILES = {
    'attitudes': 'attitudes - hate speech.csv',
    'cade': 'cade - acceptability.csv',
    'davani': 'davani - offensiveness.csv',
    'measuring': 'measuring - violence.csv',
}

mappatura_modelli = {
    'Llama-3.2-1B-Instruct': 'Llama-3.2-1B',
    'Qwen2.5-1.5B-Instruct': 'Qwen2.5-1.5B',
    'Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B',
}

mappatura_dataset = {
    'davani': 'Disentangling',
    'measuring': 'MHS',
    'attitudes': 'Attitudes',
    'cade': 'CADE',
}


def normalize_label(value):
    if pd.isna(value):
        return ''

    text = str(value).strip()
    if text == '':
        return ''

    try:
        num = float(text)
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        return text


def parse_probs(raw):
    if isinstance(raw, dict):
        return {normalize_label(key): float(value) for key, value in raw.items()}

    text = str(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = ast.literal_eval(text)

    return {normalize_label(key): float(value) for key, value in data.items()}


def extract_model_label(probs):
    return max(probs, key=probs.get)


def compute_ncs_for_label(probs, chosen_label):
    components = []
    for label, prob in probs.items():
        if label == chosen_label:
            components.append((1.0 - prob) ** 2)
        else:
            components.append(prob ** 2)
    return float(sum(components) / len(components))


def select_ghost_label(probs, human_labels):
    available_labels = list(probs.keys())
    empty_candidates = [label for label in available_labels if label not in human_labels]

    if empty_candidates:
        chosen = min(empty_candidates, key=lambda label: probs[label])
        return chosen, True

    chosen = min(available_labels, key=lambda label: probs[label])
    return chosen, False


def load_human_comment_data(dataset_key):
    human_file = os.path.join(HUMAN_DATA_DIR, HUMAN_DATASET_FILES[dataset_key])
    df = pd.read_csv(
        human_file,
        usecols=lambda col: col in {'comment_id', 'label', 'annotator_id'},
    )
    df = df.dropna(subset=['comment_id', 'label']).copy()
    df['comment_id'] = df['comment_id'].astype(str)
    df['label'] = df['label'].apply(normalize_label)

    if 'annotator_id' in df.columns:
        df = df.dropna(subset=['annotator_id']).copy()
        df = df[['comment_id', 'annotator_id', 'label']].drop_duplicates()
        total_annotators = df.groupby('comment_id')['annotator_id'].nunique()
        label_counts = df.groupby(['comment_id', 'label'])['annotator_id'].nunique()
    else:
        df = df[['comment_id', 'label']].drop_duplicates()
        total_annotators = df.groupby('comment_id').size()
        label_counts = df.groupby(['comment_id', 'label']).size()

    human_label_map = (
        df.groupby('comment_id')['label']
        .apply(lambda labels: set(label for label in labels.tolist() if label != ''))
        .to_dict()
    )

    fraction_agreement_map = {}
    for comment_id, total in total_annotators.items():
        max_same_label = label_counts.loc[comment_id].max()
        fraction_agreement_map[comment_id] = float(max_same_label / total * 100)

    return human_label_map, fraction_agreement_map


def convert_to_isolation(fraction_agreement):
    # Keep compatibility with both scales: 0-1 and 0-100.
    if fraction_agreement <= 1.0:
        return 1.0 - fraction_agreement
    return 100.0 - fraction_agreement


def process_model_file(input_path, human_label_map, fraction_agreement_map):
    df = pd.read_csv(input_path, usecols=['comment_id', 'probs'])
    df = df.dropna(subset=['comment_id', 'probs']).copy()
    df['comment_id'] = df['comment_id'].astype(str)

    per_comment = df.groupby('comment_id', as_index=False).agg(probs=('probs', 'first'))

    rows = []
    for _, row in per_comment.iterrows():
        comment_id = row['comment_id']
        probs = parse_probs(row['probs'])
        model_label = normalize_label(extract_model_label(probs))
        human_labels = human_label_map.get(comment_id, set())

        # Keep only comments where model label is never selected by human annotators.
        if model_label in human_labels:
            continue

        ghost_label, _ = select_ghost_label(probs, human_labels)
        ncs = compute_ncs_for_label(probs, ghost_label)
        fraction_agreement = fraction_agreement_map.get(comment_id)
        if fraction_agreement is None:
            continue

        rows.append(
            {
                'comment_id': comment_id,
                'ncs': ncs,
                'annotator_isolation': convert_to_isolation(fraction_agreement),
            }
        )

    return pd.DataFrame(rows)


def parse_model_and_dataset(filename):
    core = filename.replace('results_', '', 1).replace('.csv', '')
    model_name, dataset_key = core.rsplit('_', 1)
    return model_name, dataset_key


def calculate_correlation(input_path, human_maps, fraction_maps):
    filename = os.path.basename(input_path)
    model_name, dataset_key = parse_model_and_dataset(filename)

    df = process_model_file(input_path, human_maps[dataset_key], fraction_maps[dataset_key])
    if df.empty or df['ncs'].nunique() < 2 or df['annotator_isolation'].nunique() < 2:
        corr, p_value = np.nan, np.nan
    else:
        corr, p_value = pearsonr(df['ncs'], df['annotator_isolation'])

    print(
        f"{filename}  |  ghost_comments={len(df)}"
        f"  |  corr={corr if not np.isnan(corr) else 'nan'}"
        f"  |  p={p_value if not np.isnan(p_value) else 'nan'}"
    )
    return model_name, dataset_key, corr, p_value


file_csvs = sorted(
    os.path.join(ANNOTAZIONI_DIR, filename)
    for filename in os.listdir(ANNOTAZIONI_DIR)
    if filename.startswith('results_') and filename.endswith('.csv')
)

human_maps = {}
fraction_maps = {}
for dataset_key in HUMAN_DATASET_FILES:
    human_map, fraction_map = load_human_comment_data(dataset_key)
    human_maps[dataset_key] = human_map
    fraction_maps[dataset_key] = fraction_map

pearson_values = {}
p_values = {}
for file_csv in file_csvs:
    model_name, dataset_key, corr, p_value = calculate_correlation(file_csv, human_maps, fraction_maps)
    pearson_values[(mappatura_modelli[model_name], mappatura_dataset[dataset_key])] = corr
    p_values[(mappatura_modelli[model_name], mappatura_dataset[dataset_key])] = p_value

models = ['Llama-3.2-1B', 'Qwen2.5-1.5B', 'Llama-3.1-8B', 'Qwen2.5-7B']
datasets = sorted({dataset for _, dataset in pearson_values.keys()}, reverse=True)

matrix = pd.DataFrame(np.nan, index=models, columns=datasets)
p_matrix = pd.DataFrame(np.nan, index=models, columns=datasets)

for (model_name, dataset_name), corr in pearson_values.items():
    matrix.loc[model_name, dataset_name] = corr

for (model_name, dataset_name), p_value in p_values.items():
    p_matrix.loc[model_name, dataset_name] = p_value

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    square=True,
    cbar_kws={'shrink': 0.5},
    xticklabels=True,
    yticklabels=True,
    annot_kws={'color': 'black', 'fontsize': 13},
    ax=ax,
)

ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=45, ha='left', fontsize=13)
ax.set_yticks(np.arange(len(models)) + 0.5)
ax.set_yticklabels(models, rotation=0, fontsize=13)
ax.tick_params(axis='y', bottom=False, top=False)
ax.tick_params(axis='x', bottom=False, top=False)

for i in range(len(models)):
    for j in range(len(datasets)):
        p_value = p_matrix.iloc[i, j]
        if np.isnan(p_value):
            label = 'p = nan'
        elif p_value < 0.05:
            label = 'p < 0.05'
        else:
            label = 'p ≥ 0.05'
        ax.text(j + 0.5, i + 0.17, label, ha='center', va='center', fontsize=10, color='black')

plt.title('Correlation between Ghost NCS and Annotator Isolation', fontweight='bold', fontsize=20)
cbar = ax.collections[0].colorbar
cbar.set_ticks(np.linspace(-1, 1, 5))
cbar.ax.tick_params(labelsize=13)

print(pearson_values)
print(p_values)

plt.subplots_adjust(left=0.2, right=0.888, top=0.755, bottom=0.112)
os.makedirs(IMG_DIR, exist_ok=True)
output_path = os.path.join(IMG_DIR, 'correlation_between_ghost_ncs_and_isolation.pdf')
plt.savefig(output_path)
print(f'Grafico salvato in: {output_path}')
plt.show()