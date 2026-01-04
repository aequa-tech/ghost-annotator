

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
    #brier_scores = df[['fraction_agreement_Q1','fraction_agreement_Q2','fraction_agreement_Q3']].values
    social_groups = df['social_group'].values

    df_group_1 = df[df['social_group'] == 1]
    brier_scores_1 = df_group_1[['brier_score_Q1', 'brier_score_Q2', 'brier_score_Q3']].values

    df_group_0 = df[df['social_group'] == 0]
    brier_scores_0 = df_group_0[['brier_score_Q1', 'brier_score_Q2', 'brier_score_Q3']].values

    # Calcoliamo la media delle colonne brier_score_Q1, brier_score_Q2, brier_score_Q3 per il gruppo 1
    brier_scores_1 = df_group_1[['brier_score_Q1', 'brier_score_Q2', 'brier_score_Q3']].values

    # Calcolare la matrice di distanze euclidee tra tutteuclidean_distances(brier_scores)e le coppie di autori
    distanze_1 = euclidean_distances(brier_scores_1)
    # Estrarre solo le distanze tra coppie diverse (escludendo la diagonale)
    # La diagonale è composta dalle distanze tra ogni autore e se stesso, che sono 0
    # Creiamo una maschera per escluderle
    mask = np.ones_like(distanze_1, dtype=bool)
    np.fill_diagonal(mask, 0)

    # Estraiamo tutte le distanze diverse da quella tra un autore e se stesso
    distanze_1_senza_diagonale = distanze_1[mask]

    # Calcolare la distanza media
    media_1 = np.mean(distanze_1_senza_diagonale)


    distanze_0 = euclidean_distances(brier_scores_0)
    # Estrarre solo le distanze tra coppie diverse (escludendo la diagonale)
    # La diagonale è composta dalle distanze tra ogni autore e se stesso, che sono 0
    # Creiamo una maschera per escluderle
    mask = np.ones_like(distanze_0, dtype=bool)
    np.fill_diagonal(mask, 0)

    # Estraiamo tutte le distanze diverse da quella tra un autore e se stesso
    distanze_0_senza_diagonale = distanze_0[mask]

    # Calcolare la distanza media
    media_0 = np.mean(distanze_0_senza_diagonale)

    # Calcolare la matrice di distanze euclidee tra i punti del social_group_0 e quelli del social_group_1
    distanze_intergruppo = euclidean_distances(brier_scores_0, brier_scores_1)

    # Calcolare la distanza media tra i membri dei due gruppi
    media_intergruppo = np.mean(distanze_intergruppo)



    print(model_name, dataset_name, media_0,media_1,media_intergruppo)
    # Calcolare il rapporto tra la distanza intra-gruppo media e la distanza inter-gruppo media
    media_intra_gruppo = (media_0 + media_1) / 2  # media delle distanze intra-gruppo

    # Calcolare il rapporto
    rapporto = media_intra_gruppo / media_intergruppo

    # Stampiamo il rapporto
    print("Rapporto tra distanza intra-gruppo e inter-gruppo:", rapporto)
    continue
