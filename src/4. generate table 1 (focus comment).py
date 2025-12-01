import glob
import os
import pandas as pd
import numpy as np


def load_files(output_folder, modello, dataset):
    # File tipo 1 (results_{modello}_{dataset}.csv)
    file_tipo1_pattern = os.path.join(output_folder, f"results_{modello}_{dataset}.csv")
    df_tipo1_files = glob.glob(file_tipo1_pattern)

    if not df_tipo1_files:
        raise FileNotFoundError(f"File tipo 1 per il modello {modello} e dataset {dataset} non trovato.")

    df_tipo1 = pd.read_csv(df_tipo1_files[0])  # Carichiamo il primo file trovato

    # File tipo 2 ({dataset} - {fenomeno}.csv)
    file_tipo2_pattern = os.path.join("data/measuring_hatespeech", f"{dataset} - *.csv")
    tipo2_files = glob.glob(file_tipo2_pattern)

    if not tipo2_files:
        raise FileNotFoundError(f"File tipo 2 per il dataset {dataset} non trovato.")

    # Carichiamo tutti i file tipo2
    df_tipo2 = pd.concat([pd.read_csv(file) for file in tipo2_files])

    return df_tipo1, df_tipo2


def preprocess_files(df_tipo1, df_tipo2):
    # Pulizia dei dati
    df_tipo1 = df_tipo1.dropna(subset=['comment_id', 'text', 'label', 'annotator_id', 'social_group'])
    df_tipo2 = df_tipo2.dropna(subset=['comment_id', 'text', 'label', 'annotator_id', 'social_group'])

    return df_tipo1, df_tipo2


def calculate_fraction_agreement(df_tipo2):
    # Calcoliamo la percentuale di annotatori che hanno usato la stessa label
    print(df_tipo2.columns)
    total_annotators = df_tipo2.groupby('comment_id')['annotator_id'].transform('nunique')

    # Calcoliamo il numero di annotatori che hanno usato la stessa label per ciascun 'annotation_id'
    same_label_annotators = df_tipo2.groupby(['comment_id', 'label'])['annotator_id'].transform('nunique')

    # Calcoliamo la percentuale di annotatori che hanno usato la stessa label
    same_label_percentage = (same_label_annotators / total_annotators) * 100

    # Crea un nuovo DataFrame che contiene tutte le colonne originali + la colonna 'same_label_percentage'
    df_with_percentage = df_tipo2.copy()  # Creiamo una copia del DataFrame originale

    df_with_percentage['fraction_agreement'] = same_label_percentage  # Aggiungiamo la nuova colonna


    print(df_with_percentage)

    df_with_percentage = df_with_percentage[['comment_id', 'annotator_id', 'fraction_agreement']]

    return df_with_percentage


def calculate_brier_scores(df_tipo1, df_tipo2):
    # Uniamo i due dataframe sulla base di comment_id e annotator_id
    df_merged = pd.merge(df_tipo1, df_tipo2, on=['comment_id', 'annotator_id'], suffixes=('_model', ''))
    print(df_merged.columns)

    # Calcoliamo la media del Brier score per ogni comment_id
    avg_brier_score = df_merged.groupby('comment_id')['brier_score'].mean().reset_index(name='avg_brier_score')

    return df_merged, avg_brier_score


def calculate_quartiles(df_merged):
    # Calcoliamo i quartili Q1, Q2 (mediana) e Q3 per ogni comment_id
    quartiles = df_merged.groupby('comment_id')['brier_score'].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    quartiles.columns = ['comment_id', 'Q1', 'Q2', 'Q3']
    return quartiles


def aggregate_data(df_tipo1, df_tipo2, output_folder, modello, dataset):
    # Preprocessamento dei file
    df_tipo1, df_tipo2 = preprocess_files(df_tipo1, df_tipo2)

    # Calcolare frazione di accordo
    fraction_agreement = calculate_fraction_agreement(df_tipo2)

    # Calcolare Brier scores e media
    df_merged, avg_brier_score = calculate_brier_scores(df_tipo1, df_tipo2)

    # Calcolare quartili
    quartiles = calculate_quartiles(df_merged)

    # Uniamo i dati
    print("df_merged.columns")
    print(df_merged.columns)
    print("fraction_agreement.columns")
    print(fraction_agreement.columns)
    print("avg_brier_score.columns")
    print(avg_brier_score.columns)
    print("quartiles.columns")
    print(quartiles.columns)
    final_df = pd.merge(df_merged, fraction_agreement,  on=['comment_id', 'annotator_id'])
    final_df = pd.merge(final_df, avg_brier_score, on='comment_id')
    final_df = pd.merge(final_df, quartiles, on='comment_id')
    print(final_df.columns)
    # Selezioniamo le colonne richieste
    final_df = final_df[['comment_id', 'text_model', 'annotator_id', 'social_group', 'label', 'fraction_agreement',
                         'label_model', 'probs', 'brier_score', 'avg_brier_score', 'Q1', 'Q2', 'Q3']]

    # Numero di annotatori
    num_annotators = df_merged.groupby('comment_id')['annotator_id'].nunique().reset_index(name='num_annotators')
    final_df = pd.merge(final_df, num_annotators, on='comment_id')

    # Salviamo il file di output
    output_file = os.path.join(output_folder, f"step_1_{modello}_{dataset}.csv")
    final_df.to_csv(output_file, index=False)

    print(f"File {output_file} generato con successo.")


# Funzione principale per eseguire il processo per tutti i modelli e dataset
def process_all_data(output_folder):
    # Troviamo tutti i dataset disponibili nella cartella
    dataset_pattern = os.path.join("data/measuring_hatespeech", "*-*.csv")
    all_files = glob.glob(dataset_pattern)
    # Estraiamo i dataset unici
    datasets = set()
    for file in all_files:
        filename = os.path.basename(file)
        dataset_name = filename.split(' - ')[0]
        print(dataset_name)
        datasets.add(dataset_name)
    print(datasets)

    # Eseguiamo il processo per ogni dataset e modello
    for dataset in datasets:
        # Troviamo i modelli associati a ogni dataset
        modello_pattern = os.path.join(output_folder, f"results_*_{dataset}.csv")
        modello_files = glob.glob(modello_pattern)
        # Estraiamo i modelli unici
        modelli = set()
        for modello_file in modello_files:
            modello_name = os.path.basename(modello_file).split('_')[1]
            modelli.add(modello_name)

        # Eseguiamo il processo per ogni modello
        for modello in modelli:
            print(f"Processando modello {modello} e dataset {dataset}...")
            df_tipo1, df_tipo2 = load_files(output_folder, modello, dataset)
            aggregate_data(df_tipo1, df_tipo2, output_folder, modello, dataset)


# Eseguiamo il processo per tutti i dataset e modelli nella cartella di output
output_folder = 'output'  # Cambia con il percorso della tua cartella di output
process_all_data(output_folder)
