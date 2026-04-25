"""Figura 1 Bottom (versione src_2): incertezza del modello sui soli commenti ghost.

Logica:
1) Per ogni commento c si prende la label del modello come argmax delle probabilita'.
2) Si selezionano solo i commenti ghost: y_model(c) not in H(c), con H(c) insieme
    delle label usate dagli annotatori umani su c.
3) Per ogni commento ghost si calcola il Brier-style model uncertainty rispetto alla
    label scelta dal modello.

Formule:
- y_model(c) = argmax_y P_c(y)
- Ghost(c) <=> y_model(c) not in H(c)
- NCS_model(c) = (1 / K) * [ (1 - P_c(y_model(c)))^2 + sum_{y != y_model(c)} P_c(y)^2 ]

dove K e' il numero di etichette possibili nel dizionario di probabilita'.

Il boxplot mostra la distribuzione di NCS_model(c) separata per modello e dataset,
ma solo sui commenti che soddisfano la condizione Ghost(c).
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTAZIONI_DIR = os.path.join(SCRIPT_DIR, 'annotazione_modelli')
IMG_DIR = os.path.join(SCRIPT_DIR, 'img')

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


def estrai_label_modello(probs):
    return int(max(probs, key=probs.get))


def calcola_brier_score_modello(probs):
    label_model = estrai_label_modello(probs)
    conf_scores = []
    for pred, prob in probs.items():
        target = 1 if int(pred) == label_model else 0
        conf_scores.append((target - prob) ** 2)
    return float(np.mean(conf_scores))


def estrai_dati_box_plot(filepath):
    df = pd.read_csv(filepath)
    df['probs_dict'] = df['probs'].apply(json.loads)
    df['label_model'] = df['probs_dict'].apply(estrai_label_modello)
    df['brier_score_model'] = df['probs_dict'].apply(calcola_brier_score_modello)

    human_labels = df.groupby('comment_id')['label'].apply(set)
    per_comment = df.groupby('comment_id').agg(
        label_model=('label_model', 'first'),
        brier_score_model=('brier_score_model', 'first'),
    )
    per_comment['is_ghost'] = [
        per_comment.loc[comment_id, 'label_model'] not in human_labels.loc[comment_id]
        for comment_id in per_comment.index
    ]

    df_ghost = per_comment[per_comment['is_ghost']].reset_index()[['comment_id', 'brier_score_model']]

    filename = os.path.basename(filepath)
    parts = filename.split('_')
    model_name = parts[1]
    dataset_name = parts[2].split('.')[0]

    df_ghost['model_name'] = model_name
    df_ghost['dataset_name'] = dataset_name

    print(
        f"{filename}  |  ghost_texts={len(df_ghost)}/{df['comment_id'].nunique()}"
        f"  |  mean_brier={df_ghost['brier_score_model'].mean():.4f}"
    )
    return df_ghost


file_csvs = sorted(
    os.path.join(ANNOTAZIONI_DIR, filename)
    for filename in os.listdir(ANNOTAZIONI_DIR)
    if filename.startswith('results_') and filename.endswith('.csv')
)

dati_box_plot = []
for filepath in file_csvs:
    df_file = estrai_dati_box_plot(filepath)
    dati_box_plot.append(df_file)

df_box_plot = pd.concat(dati_box_plot, ignore_index=True)
df_box_plot['model_name'] = df_box_plot['model_name'].map(mappatura_modelli)
df_box_plot['dataset_name'] = df_box_plot['dataset_name'].map(mappatura_dataset)

print('\nDataFrame finale (solo ghost texts):')
print(df_box_plot.to_string(index=False))

plt.figure(figsize=(10, 6))

modelli_ordinati = ['Llama-3.2-1B', 'Qwen2.5-1.5B', 'Llama-3.1-8B', 'Qwen2.5-7B']
dataset_ordinati = sorted(df_box_plot['dataset_name'].dropna().unique(), reverse=True)
flierprops = dict(marker='o', markerfacecolor='black', markersize=1, markeredgewidth=0)

sns.boxplot(
    x='dataset_name',
    y='brier_score_model',
    hue='model_name',
    data=df_box_plot,
    hue_order=modelli_ordinati,
    palette=['#fdae61', '#abd9e9', '#d7191c', '#2c7bb6'],
    flierprops=flierprops,
    order=dataset_ordinati,
)

plt.title('Model uncertainty on Ghost Predictions Across Models and Datasets', fontweight='bold', fontsize=24)
plt.xlabel('', fontsize=18)
plt.ylabel('Model uncertainty', fontsize=18)
plt.xticks(rotation=0, fontsize=16)
plt.yticks(fontsize=16)
plt.ylim(-0.01, 0.3)

font_properties = font_manager.FontProperties(weight='bold', size=18)
plt.legend(title='', fontsize=16, title_fontproperties=font_properties, ncol=2, loc='upper left')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['bottom'].set_color('gray')
plt.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=0)

plt.tight_layout()
os.makedirs(IMG_DIR, exist_ok=True)
output_path = os.path.join(IMG_DIR, 'model_uncertainty_ghost_predictions_across_models_and_datasets.pdf')
plt.savefig(output_path)
print(f'\nGrafico salvato in: {output_path}')
plt.show()

media_per_model_dataset = (
    df_box_plot.groupby(['model_name', 'dataset_name'])['brier_score_model']
    .mean()
    .reset_index()
)

print('\nMedia dei Brier Scores del modello sui soli ghost texts per modello e dataset:')
print(media_per_model_dataset.to_string(index=False))