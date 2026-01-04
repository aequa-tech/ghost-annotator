

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
    if dataset_name not in ghost_annotator:
        ghost_annotator[dataset_name]= {}
    ghost_annotator[dataset_name][model_name]=[Q1,Q2,Q3]

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

    for ghost_dataset_name in ghost_annotator.keys():
        for ghost_model_name in ghost_annotator[ghost_dataset_name].keys():

            if ghost_dataset_name+' '+ghost_model_name not in head_map_data:
                head_map_data[ghost_dataset_name+' '+ghost_model_name] = {}

            distances_0=[]
            distances_1=[]
            for i in range(0,len(brier_scores)):
                brier_score = np.array(brier_scores[i])
                ghost_annotator_vector = np.array(ghost_annotator[ghost_dataset_name][ghost_model_name])
                distance = cosine_distances([brier_score], [ghost_annotator_vector])[0][0]

                if social_groups[i]==0:
                    # Calcola la distanza Euclidea tra i due vettori (brier_score e ghost_annotator_vector)
                    distances_0.append(distance)
                else:
                    distances_1.append(distance)

            avg_distance_0 = np.mean(distances_0)
            avg_distance_1 = np.mean(distances_1)
            #print(avg_distance_0,avg_distance_1,avg_distance_0 / float(avg_distance_1))
            if dataset_name not in head_map_data[ghost_dataset_name+' '+ghost_model_name]:
                head_map_data[ghost_dataset_name+' '+ghost_model_name][dataset_name]={}
            head_map_data[ghost_dataset_name+' '+ghost_model_name][dataset_name][model_name] = (float(avg_distance_1)-avg_distance_0 ) / (float(avg_distance_1)+avg_distance_0 )


print(head_map_data)
def plot_heatmap(data_dict, title):
    # Convertiamo il dizionario in una matrice 2D per la heatmap
    data_matrix = np.array([[v for k, v in subdict.items()] for subdict in data_dict.values()]).T

    # Crea una heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_matrix, annot=True, cmap='coolwarm', cbar=True, xticklabels=list(data_dict.keys()), yticklabels=list(data_dict.values())[0].keys(),vmin=-1, vmax=1)
    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel('Datasets')
    plt.tight_layout()
    plt.savefig(f"{title}.png")  # Sostituisci il nome del file come preferisci
    #plt.show()

# Creare una heatmap per ogni "ghost" annotator (chiave di dataset e modello)
for key, value in head_map_data.items():

    plot_heatmap(value, f'Ghost annotator profile {key}')

# Creare una nuova struttura per la media dei valori di tutte le heatmap
mean_heatmap_data = {}

# Per ogni dataset e modello, calcoliamo la media dei valori di tutte le heatmap
for key, value in head_map_data.items():
    for dataset_name, model_data in value.items():
        if dataset_name not in mean_heatmap_data:
            mean_heatmap_data[dataset_name] = {}

        for model_name, val in model_data.items():
            if model_name not in mean_heatmap_data[dataset_name]:
                mean_heatmap_data[dataset_name][model_name] = []

            mean_heatmap_data[dataset_name][model_name].append(val)
# Calcoliamo la media per ogni quadrato (i, j)
final_heatmap_data = {}

for dataset_name, model_data in mean_heatmap_data.items():
    final_heatmap_data[dataset_name] = {}
    for model_name, values in model_data.items():
        final_heatmap_data[dataset_name][model_name] = np.mean(values)

plot_heatmap(final_heatmap_data, "Ghost profile Average Heatmap of All Models All datasets.png")