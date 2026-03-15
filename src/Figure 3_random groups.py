

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

from R_utilities import mappatura_dataset

# Cartella contenente i file CSV
cartella_output = 'output_def'

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
    dataset_name = mappatura_dataset[dataset_name]

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



import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

num = np.random.default_rng(42)

# accumulate sums and counts
heatmap_sum = {}
heatmap_count = {}

for seed in num.integers(0,1000000,size=20):

    rnd = np.random.default_rng(seed)
    head_map_data = {}

    for file_csv in file_csvs:
        print("\n",file_csv)

        df = pd.read_csv(os.path.join(cartella_output, file_csv))

        groups = rnd.integers(0,2,size=len(df))
        df['social_group'] = groups

        model_name = file_csv.split('_')[2]
        dataset_name = file_csv.split('_')[-1].split(".")[0]

        print(model_name,dataset_name)

        dataset_name = mappatura_dataset[dataset_name]

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

                for i in range(len(brier_scores)):

                    brier_score = np.array(brier_scores[i])
                    ghost_vector = np.array(ghost_annotator[ghost_model_name][ghost_dataset_name])

                    distance = cosine_distances([brier_score], [ghost_vector])[0][0]

                    if social_groups[i]==0:
                        distances_0.append(distance)
                    else:
                        distances_1.append(distance)

                distances_0_sorted = sorted(distances_0)[:10]
                distances_1_sorted = sorted(distances_1)[:10]

                avg_distance_0 = np.mean(distances_0_sorted)
                avg_distance_1 = np.mean(distances_1_sorted)

                value = abs(avg_distance_1-avg_distance_0) / (avg_distance_1+avg_distance_0)

                if ghost_dataset_name not in head_map_data[ghost_model_name]:
                    head_map_data[ghost_model_name][ghost_dataset_name] = {}

                head_map_data[ghost_model_name][ghost_dataset_name][dataset_name] = value

    # -------- accumulate results --------

    for model in head_map_data:
        for row in head_map_data[model]:
            for col in head_map_data[model][row]:

                value = head_map_data[model][row][col]

                heatmap_sum.setdefault(model, {}).setdefault(row, {}).setdefault(col, 0)
                heatmap_count.setdefault(model, {}).setdefault(row, {}).setdefault(col, 0)

                heatmap_sum[model][row][col] += value
                heatmap_count[model][row][col] += 1


# -------- compute final average --------

heatmap_avg = {}

for model in heatmap_sum:

    heatmap_avg[model] = {}

    for row in heatmap_sum[model]:

        heatmap_avg[model][row] = {}

        for col in heatmap_sum[model][row]:

            heatmap_avg[model][row][col] = (
                heatmap_sum[model][row][col] /
                heatmap_count[model][row][col]
            )


print("\nFINAL AVERAGED HEATMAP")
print(heatmap_avg)


# -------- Heatmap visualization function --------

def generate_heatmap(model_name, heat_map):

    model_data = heat_map[model_name]

    datasets = list(model_data.keys())
    datasets.sort(reverse=True)

    data_matrix = []

    for row in datasets:
        data_matrix.append([model_data[row][col] for col in datasets])

    data_matrix = data_matrix[::-1]

    plt.figure(figsize=(8,6))

    ax = sns.heatmap(
        data_matrix,
        annot=True,
        cmap='coolwarm',
        xticklabels=datasets,
        yticklabels=datasets[::-1],
        vmin=0,
        vmax=1,
        fmt='.2f',
        cbar_kws={"shrink":0.5},
        annot_kws={"color":"black","fontsize":13},
        square=True
    )

    plt.title(f"Group bias with {model_name}",fontweight='bold',fontsize=20)

    # Fix colorbar ticks
    cbar = ax.collections[0].colorbar
    ticks = np.linspace(0,1,3)  # 0, 0.2, 0.4, 0.6, 0.8, 1
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.1f}" for t in ticks])
    cbar.ax.tick_params(labelsize=13)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    ax.set_ylabel('Ghost Annotator Profile',fontsize=13)
    ax.set_xlabel('User Profile',fontsize=13)

    plt.savefig(f"img/Gender_bias_averaged_{model_name}.pdf")

    plt.show()
    plt.close()

# -------- plot averaged heatmaps --------

for model in heatmap_avg:
    generate_heatmap(model, heatmap_avg)