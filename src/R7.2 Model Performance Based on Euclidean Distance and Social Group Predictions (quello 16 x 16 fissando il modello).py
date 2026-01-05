

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

            avg_distance_0 = np.mean(distances_0)
            avg_distance_1 = np.mean(distances_1)
            #print(avg_distance_0,avg_distance_1,avg_distance_0 / float(avg_distance_1))
            if ghost_dataset_name not in head_map_data[ghost_model_name]:
                head_map_data[ghost_model_name][ghost_dataset_name]={}
            head_map_data[ghost_model_name][ghost_dataset_name][dataset_name] = (float(avg_distance_1)-avg_distance_0 ) / (float(avg_distance_1)+avg_distance_0 )


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
    plt.savefig(f"img/{title}.png")  # Sostituisci il nome del file come preferisci
    plt.show()

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
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Supponiamo che `heat_map` sia il dizionario che hai fornito.
heat_map = {
    'Qwen2.5-7B-Instruct': {
        'measuring': {'measuring': np.float64(-0.00681610080785954), 'davani': np.float64(0.0015770813066581142),
                   'attitudes': np.float64(-0.18117128353522513), 'cade': np.float64(0.14924862302621011)},
        'davani': {'measuring': np.float64(-0.0059482663187883445), 'davani': np.float64(0.0003959240205685532),
                   'attitudes': np.float64(-0.25347405815719526), 'cade': np.float64(0.127542316342511)},
        'attitudes': {'measuring': np.float64(-0.007527073023580467), 'davani': np.float64(0.002490106878937915),
                      'attitudes': np.float64(-0.14016969565922066), 'cade': np.float64(0.1647236106159611)},
        'cade': {'measuring': np.float64(-0.005780833195289886), 'davani': np.float64(0.00044536404970565123),
                 'attitudes': np.float64(-0.25155776028995586), 'cade': np.float64(0.124421797034258)}
    },
    'Llama-3.2-1B-Instruct': {
        'attitudes': {'attitudes': np.float64(0.07804802428310889), 'measuring': np.float64(-0.03415396827229657),
                      'davani': np.float64(0.024188070511979902), 'cade': np.float64(0.07793022431173054)},
        'measuring': {'attitudes': np.float64(0.07677783816662816), 'measuring': np.float64(-0.034489786851672755),
                   'davani': np.float64(0.02807461668110284), 'cade': np.float64(0.12689331761817543)},
        'davani': {'attitudes': np.float64(0.07817634739249339), 'measuring': np.float64(-0.034314567721204815),
                   'davani': np.float64(0.0231994486769855), 'cade': np.float64(0.039708858898281525)},
        'cade': {'attitudes': np.float64(0.06828619991992306), 'measuring': np.float64(-0.03044168334898818),
                 'davani': np.float64(0.03197375352659472), 'cade': np.float64(0.2041503125507743)}
    },
    'Llama-3.1-8B-Instruct': {
        'cade': {'cade': np.float64(-0.16917878773450792), 'measuring': np.float64(-0.01819709103976764),
                 'attitudes': np.float64(0.03719246083057927), 'davani': np.float64(-0.014721940442076531)},
        'measuring': {'cade': np.float64(-0.14316142674447638), 'measuring': np.float64(-0.01908048850897995),
                   'attitudes': np.float64(0.03193145560640635), 'davani': np.float64(-0.0153660022196154)},
        'attitudes': {'cade': np.float64(-0.17095647745580622), 'measuring': np.float64(-0.02014230925029515),
                      'attitudes': np.float64(0.038803138225264135), 'davani': np.float64(-0.014252633995678643)},
        'davani': {'cade': np.float64(-0.1457032591401882), 'measuring': np.float64(-0.02013127656179256),
                   'attitudes': np.float64(0.033475245767827726), 'davani': np.float64(-0.014978463068321773)}
    },
    'Qwen2.5-1.5B-Instruct': {
        'davani': {'davani': np.float64(-0.021328137094703506), 'attitudes': np.float64(0.011172845152355402),
                   'cade': np.float64(-0.24213542342349018), 'measuring': np.float64(0.04814618304407382)},
        'attitudes': {'davani': np.float64(-0.02306951042040499), 'attitudes': np.float64(-0.0027450971932097322),
                      'cade': np.float64(0.03204824373089993), 'measuring': np.float64(0.005038808094207454)},
        'cade': {'davani': np.float64(-0.030070942764749056), 'attitudes': np.float64(0.00799871218794889),
                 'cade': np.float64(-0.1289714210617606), 'measuring': np.float64(0.03753680524966773)},
        'measuring': {'davani': np.float64(-0.029422431020461576), 'attitudes': np.float64(0.00620786112336076),
                   'cade': np.float64(-0.07418618959608198), 'measuring': np.float64(0.027875125451901288)}
    }
}


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
    plt.savefig(f"img/Gender bias for Ghost Annotators profiled with {model_name}.png")  # Sostituisci il nome del file come preferisci

    # Mostra la heatmap
    plt.show()


# Genera la heatmap per ciascun modello
for model in heat_map:
    generate_heatmap(model, heat_map)