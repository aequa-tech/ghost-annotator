import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import glob

# Cartella contenente i file CSV
cartella_output = '../output_def'


# Funzione per estrarre la media del Brier score per ogni modello e dataset
def calcola_brier_score(file_csv):
    df = pd.read_csv(file_csv)
    model_name = file_csv.split('/')[-1].split('_')[2]
    dataset_name = file_csv.split('_')[-1].split(".")[0]

    # Calcolare la media del brier_score_model per ogni modello e dataset
    media_brier_score = df['brier_score_model'].mean()
    print(model_name,dataset_name,media_brier_score)
    return model_name, dataset_name, media_brier_score


# Funzione per calcolare la percentuale di allucinazioni per ogni modello e dataset
def calcola_percentuale_allucinazioni(file_csv):
    df = pd.read_csv(file_csv)
    print(file_csv)
    model_name = file_csv.split('/')[-1].split('_')[2]
    dataset_name = file_csv.split('_')[-1].split(".")[0]

    # Calcolare la percentuale di allucinazioni
    n_commenti_predetti = df['comment_id'].nunique()
    dataset_base_file = glob.glob(f"../data/measuring_hatespeech/{dataset_name} - *")[0]
    df_base = pd.read_csv(dataset_base_file)
    df_base = df_base.dropna(subset=['text', 'label', 'annotator_id', 'social_group'])
    n_commenti = df_base['comment_id'].nunique()

    if n_commenti_predetti == 0:
        percentuale = 0
    else:
        percentuale = (n_commenti - n_commenti_predetti) / n_commenti * 100
    print(model_name,dataset_name,percentuale)

    return model_name, dataset_name, percentuale


# Lista dei file CSV nella cartella di output
file_csvs = [f for f in os.listdir(cartella_output) if f.startswith('step_1_') and f.endswith('.csv')]

# Dizionari per memorizzare i dati
brier_scores = {}
allucinazioni = {}

# Estrazione dei dati di brier_score e percentuale di allucinazioni
for file_csv in file_csvs:
    # Estrai media Brier score
    model_name, dataset_name, brier_score = calcola_brier_score(os.path.join(cartella_output, file_csv))
    brier_scores[(model_name, dataset_name)] = brier_score

    # Estrai percentuale di allucinazioni
    model_name, dataset_name, percentuale = calcola_percentuale_allucinazioni(os.path.join(cartella_output, file_csv))
    allucinazioni[(model_name, dataset_name)] = percentuale

# Creazione della matrice dei dati per la heatmap
models = sorted(set([key[0] for key in brier_scores.keys()]))
datasets = sorted(set([key[1] for key in brier_scores.keys()]))

# Creazione della matrice dei Brier score
matrix_brier = pd.DataFrame(np.zeros((len(models), len(datasets))), index=models, columns=datasets)
matrix_allucinazioni = pd.DataFrame(np.zeros((len(models), len(datasets))), index=models, columns=datasets)

# Popolamento delle matrici
print(brier_scores)
for (model, dataset), brier_score in brier_scores.items():
    print(model,dataset)
    matrix_brier.loc[model, dataset] = brier_score

print(allucinazioni)
for (model, dataset), percentuale in allucinazioni.items():
    print(model,dataset)
    matrix_allucinazioni.loc[model, dataset] = percentuale

# Calcolare la correlazione tra la media del Brier score e la percentuale di allucinazioni
print(matrix_brier)
print(matrix_allucinazioni)
# Appiattire entrambe le matrici in vettori unidimensionali
# Appiattire i DataFrame in array NumPy
brier_flat = matrix_brier.to_numpy().flatten()
hallucination_flat = matrix_allucinazioni.to_numpy().flatten()
# Calcolare la correlazione di Pearson e il p-value tra i due vettori
correlation, p_value = pearsonr(brier_flat, hallucination_flat)

print(f"Correlazione tra Brier e Hallucination (tutti i valori): {correlation:.4f}")
print(f"p-value: {p_value:.4f}")