

import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import glob

from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import cosine_distances

# Cartella contenente i file CSV
cartella_output = '../output_def'



# Lista dei file CSV nella cartella di output
file_csvs = [f for f in os.listdir(cartella_output) if f.startswith('step_2_') and f.endswith('.csv')]

# Dizionari per memorizzare i dati

right_values={}
p_values={}
# Estrazione dei dati di brier_score e percentuale di allucinazioni
for file_csv in file_csvs:
    print("\n",file_csv)
    df = pd.read_csv(os.path.join(cartella_output, file_csv))
    model_name = file_csv.split('_')[2]
    dataset_name = file_csv.split('_')[-1].split(".")[0]
    print(model_name,dataset_name)
    df_filtrato = df[df['social_group'].isin([0,1])]
    p_0=df['social_group'].value_counts(normalize=True)[0]#*100
    p_1=df['social_group'].value_counts(normalize=True)[1]#*100

    #print(df.columns)
    # Creiamo una matrice di feature solo con i brier_score (Q1, Q2, Q3)
    brier_scores = df[['brier_score_Q1', 'brier_score_Q2', 'brier_score_Q3']].values
    #brier_scores = df[['fraction_agreement_Q1','fraction_agreement_Q2','fraction_agreement_Q3']].values
    social_groups = df['social_group'].values

    # Calcolare la matrice di distanze euclidee tra tutteuclidean_distances(brier_scores)e le coppie di autori
    distanze = euclidean_distances(brier_scores)
    # Convertiamo la matrice di distanze in un DataFrame per una visualizzazione più facile
    #distanze_df = pd.DataFrame(distanze, index=df['annotator_id'], columns=df['annotator_id'])
    #p_0=p_1=0.5
    distanze_0=distanze*p_0
    distanze_1=distanze*p_1
    media_0 = distanze_0.mean(axis=0)
    media_1 = distanze_1.mean(axis=0)

    right=0
    wrong=0
    for i in range(0,len(social_groups)):
        if social_groups[i]==1 and media_1[i]>media_0[i]:
            right+=1
        if  social_groups[i]==0 and media_0[i]>media_1[i]:
            right+=1
        else:
            wrong+=1
    print(right,wrong,right/(right+wrong), p_0,p_1)
    right_values[(model_name, dataset_name)]=(right/(right+wrong)-max([p_0,p_1]))
    p_values[(model_name, dataset_name)]=round(right/(right+wrong)*100,2)
models = []
for key in right_values.keys():
    if key[0] not in models:
        models.append(key[0])

datasets = []
for key in right_values.keys():
    if key[1] not in datasets:
        datasets.append(key[1])

# Creazione della matrice vuota per tau e p-value
matrix = pd.DataFrame(np.zeros((len(models), len(datasets))), index=models, columns=datasets)
p_matrix = pd.DataFrame(np.ones((len(models), len(datasets))), index=models, columns=datasets)

# Popolamento della matrice con i valori di tau e p-value
for (var1, var2), tau in right_values.items():
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
        # if i < j:  # Solo triangolo superiore
        p_val = p_matrix.iloc[i, j]
        text_color = "black"

        ax.text(j + 0.5, i + 0.17, f"{p_val}%", ha='center', va='center', fontsize=8, color=text_color)

plt.title('', fontweight='bold')

print(right_values)
print(p_values)
# Spazio regolato
plt.subplots_adjust(left=0.2, right=0.9, top=0.823, bottom=0.2)

plt.show()