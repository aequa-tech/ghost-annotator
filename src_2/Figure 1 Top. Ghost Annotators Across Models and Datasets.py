"""Figura 1 Top (versione src_2): percentuale di predizioni ghost per modello e dataset.

Idea del grafico:
- Per ogni commento c, il modello produce una label y_model(c) (argmax delle probabilita').
- Gli umani hanno un insieme di label H(c) raccolte da tutte le annotazioni del commento.
- Un commento e' ghost se y_model(c) non appartiene a H(c).

Definizioni formali:
- y_model(c) = argmax_y P_c(y)
- I_ghost(c) = 1 se y_model(c) not in H(c), altrimenti 0
- Ghost%(m, d) = 100 * [sum_c I_ghost(c)] / N_{m,d}

dove:
- m = modello, d = dataset
- N_{m,d} = numero di commenti unici nel file results_{m}_{d}.csv

Nel grafico a barre viene mostrato Ghost%(m, d) per ogni coppia modello-dataset.
"""

import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTAZIONI_DIR = os.path.join(SCRIPT_DIR, "annotazione_modelli")
IMG_DIR = os.path.join(SCRIPT_DIR, "", "img")

mappatura_modelli = {
    'Llama-3.2-1B-Instruct': 'Llama-3.2-1B',
    'Qwen2.5-1.5B-Instruct': 'Qwen2.5-1.5B',
    'Llama-3.1-8B-Instruct': 'Llama-3.1-8B',
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B'
}

mappatura_dataset = {
    'davani': 'Disentangling',
    'measuring': 'MHS',
    'attitudes': 'Attitudes',
    'cade': 'CADE'
}


def calcola_percentuale_predizioni(filepath):
    df = pd.read_csv(filepath)

    # Estrai label_model come argmax delle probabilità
    df['probs_dict'] = df['probs'].apply(json.loads)
    df['label_model'] = df['probs_dict'].apply(lambda d: int(max(d, key=d.get)))

    # Per ogni comment_id, insieme delle etichette umane
    human_labels = df.groupby('comment_id')['label'].apply(set)

    # Predizione del modello per ogni comment_id (è unica per commento)
    model_pred = df.groupby('comment_id')['label_model'].first()

    # Conta i commenti in cui la predizione del modello NON è tra le etichette umane
    ghost_count = sum(
        model_pred[cid] not in human_labels[cid]
        for cid in model_pred.index
    )

    n_comments = df['comment_id'].nunique()
    percentuale = ghost_count / n_comments * 100 if n_comments > 0 else 0

    # Estrai nome modello e dataset dal nome file
    filename = os.path.basename(filepath)
    parts = filename.split('_')   # ['results', 'ModelName', 'dataset.csv']
    model_name = parts[1]
    dataset_name = parts[2].split('.')[0]

    print(f"{filename}  |  model={model_name}  dataset={dataset_name}  ghost={ghost_count}/{n_comments}  ({percentuale:.1f}%)")
    return model_name, dataset_name, percentuale


# Leggi tutti i file results_*.csv in annotazione_modelli/
file_csvs = sorted([
    os.path.join(ANNOTAZIONI_DIR, f)
    for f in os.listdir(ANNOTAZIONI_DIR)
    if f.startswith('results_') and f.endswith('.csv')
])

dati_per_grafico = []
for filepath in file_csvs:
    model_name, dataset_name, percentuale = calcola_percentuale_predizioni(filepath)
    dati_per_grafico.append([model_name, dataset_name, percentuale])

df_risultati = pd.DataFrame(dati_per_grafico, columns=['model_name', 'dataset_name', 'percentuale'])

# Rimappa nomi
df_risultati['model_name'] = df_risultati['model_name'].map(mappatura_modelli)
df_risultati['dataset_name'] = df_risultati['dataset_name'].map(mappatura_dataset)

print("\nDataFrame finale:")
print(df_risultati.to_string(index=False))

# Grafico
modelli_ordinati = ['Llama-3.2-1B', 'Qwen2.5-1.5B', 'Llama-3.1-8B', 'Qwen2.5-7B']

plt.figure(figsize=(10, 6))
sns.barplot(
    x='dataset_name',
    y='percentuale',
    hue='model_name',
    data=df_risultati,
    hue_order=modelli_ordinati,
    palette=['#fdae61', '#abd9e9', '#d7191c', '#2c7bb6'],
    zorder=3
)

plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
plt.ylim(0, 105)

font_properties = font_manager.FontProperties(weight='bold', size=18)
plt.legend(title="", fontsize=16, title_fontproperties=font_properties, ncol=1, loc='upper right')
plt.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(True)
plt.gca().spines['left'].set_color('gray')
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['bottom'].set_color('gray')

plt.title('Ghost Predictions Across Models and Datasets', fontweight='bold', fontsize=24)
plt.xlabel('', fontsize=18)
plt.ylabel('Percentage of Ghost Predictions (%)', fontsize=18)
plt.xticks(rotation=0, fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

os.makedirs(IMG_DIR, exist_ok=True)
output_path = os.path.join(IMG_DIR, "ghost_annotators_across_models_and_datasets.pdf")
plt.savefig(output_path)
print(f"\nGrafico salvato in: {output_path}")
plt.show()

# Stampa media per modello e dataset
media = df_risultati.groupby(['model_name', 'dataset_name'])['percentuale'].mean().reset_index()
print("\nMedia delle percentuali ghost per modello e dataset:")
print(media.to_string(index=False))
