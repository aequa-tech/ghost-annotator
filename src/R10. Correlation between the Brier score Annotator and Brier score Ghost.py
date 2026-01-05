import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import os
import json
cartella_output = '../output_def'  # Sostituisci con il percorso corretto

def calcola_correlazione(file_csv):
    # Leggi il file CSV
    df = pd.read_csv(file_csv)

    # Raggruppa i dati per comment_id e annotator_id
    annotazioni_umanes = df[['comment_id', 'label']].drop_duplicates()

    # Unisce con le predizioni del modello per calcolare la percentuale
    predizioni_modello = df[['comment_id', 'label_model']].drop_duplicates()

    # Raggruppiamo le etichette degli annotatori per ogni comment_id
    annotazioni_per_comment_id = annotazioni_umanes.groupby('comment_id')['label'].apply(set).reset_index()

    # Aggiungiamo le etichette degli annotatori al DataFrame delle predizioni
    df_completo = pd.merge(predizioni_modello, annotazioni_per_comment_id, on='comment_id', how='left')

    comment_id_with_ghost=[]
    for index, row in df_completo.iterrows():

        if row['label_model'] not in row['label']:
            print(row)
            comment_id_with_ghost.append(row['comment_id'])

    df = df[~df['comment_id'].isin(comment_id_with_ghost)]
    # Calcola la percentuale di predizioni "mai selezionate" rispetto al totale delle predizioni per ogni modello e dataset
    model_name, dataset_name = file_csv.split('/')[-1].split('_')[2], file_csv.split('_')[-1].split(".")[0]

    # Funzione per estrarre il valore massimo da una stringa JSON
    def extract_max_prob(probs_str):
        probs_dict = json.loads(probs_str)  # Converti la stringa in un dizionario
        return max(probs_dict.values())  # Restituisci il valore massimo

    # Crea una nuova colonna 'max_prob' con il valore massimo estratto da 'probs'
    df['brier_ghost'] = df['probs'].apply(extract_max_prob)
    corr, p_value = pearsonr(df['brier_score'], df['brier_ghost'])
    return model_name, dataset_name, corr, p_value

# Lista dei file CSV nella cartella di output
file_csvs = [f for f in os.listdir(cartella_output) if f.startswith('step_1_') and f.endswith('.csv')]

# Definizione delle variabili e delle correlazioni di Pearson
pearsonr_values = {}
p_values = {}

# Calcola la percentuale per ogni file CSV
for file_csv in file_csvs:
    model_name, dataset_name, correlazione, p_value = calcola_correlazione(os.path.join(cartella_output, file_csv))

    pearsonr_values[(model_name, dataset_name)] = correlazione
    p_values[(model_name, dataset_name)] = p_value

# Lista ordinata delle variabili
models = []
for key in p_values.keys():
    if key[0] not in models:
        models.append(key[0])

datasets = []
for key in p_values.keys():
    if key[1] not in datasets:
        datasets.append(key[1])

datasets.sort(reverse=True)
models = ['Llama-3.2-1B-Instruct', 'Qwen2.5-1.5B-Instruct', 'Llama-3.1-8B-Instruct','Qwen2.5-7B-Instruct']  # Inserisci l'ordine che desideri

# Creazione della matrice vuota per tau e p-value
matrix = pd.DataFrame(np.zeros((len(models), len(datasets))), index=models, columns=datasets)
p_matrix = pd.DataFrame(np.ones((len(models), len(datasets))), index=models, columns=datasets)

# Popolamento della matrice con i valori di tau e p-value
for (var1, var2), tau in pearsonr_values.items():
    matrix.loc[var1, var2] = tau

for (var1, var2), p_val in p_values.items():
    p_matrix.loc[var1, var2] = p_val

# Creazione della heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, square=True,
            cbar_kws={"shrink": 0.5}, xticklabels=True, yticklabels=True, annot_kws={"color": "black"}, ax=ax)

# Spostare le etichette della x in alto
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
plt.xticks(rotation=45, ha='left')

# Aggiungere le etichette delle righe correttamente
ax.set_yticks(np.arange(len(models)) + 0.5)
ax.set_yticklabels(models, rotation=0)

# Rimozione delle tacche dell'asse Y
ax.tick_params(axis='y', bottom=False, top=False)
ax.tick_params(axis='x', bottom=False, top=False)

# Aggiungere i p-value sopra i quadrati della heatmap
for i in range(len(models)):
    for j in range(len(datasets)):
        #if i < j:  # Solo triangolo superiore
        p_val = p_matrix.iloc[i, j]
        if p_val < 0.05:  # Mostra solo p-value significativi
            text_color = "black"
            ax.text(j + 0.5, i + 0.17, f"p < 0.05", ha='center', va='center', fontsize=8, color=text_color)
        else:
            text_color = "black"
            ax.text(j + 0.5, i + 0.17, f"p ≥ 0.05", ha='center', va='center', fontsize=8, color=text_color)

plt.title('Correlation between the Brier score and Ghost', fontweight='bold')

print(pearsonr_values)
print(p_values)
# Spazio regolato
plt.subplots_adjust(left=0.2, right=0.9, top=0.823, bottom=0.2)
plt.savefig(
    f"img/correlation_between_uncertainty_and_ghost.png")  # Sostituisci il nome del file come preferisci

plt.show()
