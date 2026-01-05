

import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import glob

from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances
import json
# Cartella contenente i file CSV
cartella_output = '../output_def'

file_csvs = [f for f in os.listdir(cartella_output) if f.startswith('step_2_') and f.endswith('.csv')]
print(file_csvs)
"""
ghost_annotator contiene la rappresentazione dei ghost annotator di ogni modello per ogni dataset
    ghost_annotator[model_name+' '+dataset_name]=[Q1,Q2,Q3]

"""

ghost_annotator={}
for file_csv in file_csvs:
    print("\n", file_csv)
    df = pd.read_csv(os.path.join(cartella_output, file_csv.replace('step_2_', 'step_1_')))
    model_name = file_csv.split('/')[-1].split('_')[2]
    dataset_name = file_csv.split('_')[-1].split(".")[0]
    print(model_name, dataset_name)

    annotazioni_umanes = df[['comment_id', 'label']].drop_duplicates()
    predizioni_modello = df[['comment_id', 'label_model']].drop_duplicates()

    annotazioni_per_comment_id = annotazioni_umanes.groupby('comment_id')['label'].apply(set).reset_index()

    df_completo = pd.merge(predizioni_modello, annotazioni_per_comment_id, on='comment_id', how='left')

    comment_id_with_ghost=[]
    for index, row in df_completo.iterrows():

        if row['label_model'] not in row['label']:
            comment_id_with_ghost.append(row['comment_id'])

    df = df[~df['comment_id'].isin(comment_id_with_ghost)]

    def extract_max_prob(probs_str):
        probs_dict = json.loads(probs_str)  # Converti la stringa in un dizionario
        return max(probs_dict.values())  # Restituisci il valore massimo

    # Crea una nuova colonna 'max_prob' con il valore massimo estratto da 'probs'
    df['brier_ghost'] = df['probs'].apply(extract_max_prob)
    Q1 = df['brier_ghost'].quantile(0.25)  # Primo quartile (25%)
    Q2 = df['brier_ghost'].quantile(0.50)  # Mediana (50%)
    Q3 = df['brier_ghost'].quantile(0.75)
    if model_name not in ghost_annotator:
        ghost_annotator[model_name]= {}
    ghost_annotator[model_name][dataset_name]=[Q1,Q2,Q3]

head_map_data={}
for file_csv in file_csvs:
    print("\n",file_csv)
    df = pd.read_csv(os.path.join(cartella_output, file_csv))
    model_name = file_csv.split('_')[2]
    dataset_name = file_csv.split('_')[-1].split(".")[0]
    print(model_name,dataset_name)

    df_filtrato = df[df['social_group'].isin([0,1])]
    brier_scores = df[['brier_score_Q1', 'brier_score_Q2', 'brier_score_Q3']].values
    social_groups = df['social_group'].values

    for ghost_model_name in ghost_annotator.keys():
        if ghost_model_name != model_name:
            continue
        for ghost_dataset_name in ghost_annotator[ghost_model_name].keys():

            if ghost_model_name not in head_map_data:
                head_map_data[ghost_model_name] = {}

            distances_0=[]
            distances_1=[]
            for i in range(0,len(brier_scores)):
                brier_score = np.array(brier_scores[i])
                ghost_annotator_vector = np.array(ghost_annotator[ghost_model_name][ghost_dataset_name])
                distance = cosine_distances([brier_score], [ghost_annotator_vector])[0][0]

                if social_groups[i]==0:
                    # Calcola la distanza Euclidea tra i due vettori (brier_score e ghost_annotator_vector)
                    distances_0.append(distance)
                else:
                    distances_1.append(distance)
            distances_0_sorted = sorted(distances_0)[:]
            distances_1_sorted = sorted(distances_1)[:]

            avg_distance_0 = np.mean(distances_0_sorted)
            avg_distance_1 = np.mean(distances_1_sorted)

            #print(avg_distance_0,avg_distance_1,avg_distance_0 / float(avg_distance_1))
            if ghost_dataset_name not in head_map_data[ghost_model_name]:
                head_map_data[ghost_model_name][ghost_dataset_name]={}
            head_map_data[ghost_model_name][ghost_dataset_name][dataset_name] = (float(avg_distance_1)-avg_distance_0 ) / (float(avg_distance_1)+avg_distance_0 )


print(head_map_data)

# Funzione per generare la heatmap per un dato modello
def generate_heatmap(model_name, heat_map):
    # Otteniamo i dati relativi a ciascun modello
    model_data = heat_map[model_name]

    # Creiamo una matrice di valori per la heatmap
    data_matrix = []
    datasets = list(model_data.keys())
    datasets.sort(reverse=True)

    for row in datasets:
        data_matrix.append([model_data[row][col] for col in datasets])
    data_matrix = data_matrix[::-1]  # Inverti la matrice dei dati
    # Creazione della figura e della heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        data_matrix, annot=True, cmap='coolwarm', xticklabels=datasets, yticklabels=datasets[::-1],
        vmin=-1, vmax=1, fmt='.2f', cbar_kws={'label': 'Valore'}, square=True
    )

    # Aggiungi il titolo
    plt.title(f"Gender bias for Ghost Annotators profiled with {model_name}",fontweight='bold')

    # Aggiungi le etichette agli assi
    ax.set_ylabel('Ghost Annotator Datasets Profiling', fontsize=12)
    ax.set_xlabel('User Dataset Profiling', fontsize=12)
    plt.savefig(f"img/Gender bias for Ghost Annotators profiled with {model_name} first -.png")  # Sostituisci il nome del file come preferisci

    # Mostra la heatmap
    plt.show()

# Genera la heatmap per ciascun modello
for model in head_map_data:
    generate_heatmap(model, head_map_data)

