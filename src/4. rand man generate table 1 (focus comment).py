import glob
import json
import os
import pandas as pd
import numpy as np


def load_files(output_folder, modello, dataset):
    # File tipo 1 (results_{modello}_{dataset}.csv)
    file_tipo_result_pattern = os.path.join(output_folder, f"results_{modello}_{dataset}.csv")
    df_tipo_result_files = glob.glob(file_tipo_result_pattern)
    if not df_tipo_result_files:
        raise FileNotFoundError(f"File tipo 1 per il modello {modello} e dataset {dataset} non trovato.")

    df_tipo_model = pd.read_csv(df_tipo_result_files[0])  # Carichiamo il primo file trovato

    # Funzione per ottenere la chiave con il valore massimo nel dizionario
    def get_max_key(prons_dict):
        # Trova la chiave con il valore massimo
        return max(prons_dict, key=prons_dict.get)

    # Usa apply per aggiornare la colonna 'label'
    df_tipo_model['label'] = df_tipo_model['probs'].apply(lambda prons_dict: get_max_key(eval(prons_dict)))

    # File tipo 2 ({dataset} - {fenomeno}.csv)
    file_tipo_annotator_pattern = os.path.join("data/measuring_hatespeech", f"{dataset} - *.csv")
    tipo2_files = glob.glob(file_tipo_annotator_pattern)

    if not tipo2_files:
        raise FileNotFoundError(f"File tipo 2 per il dataset {dataset} non trovato.")

    # Carichiamo tutti i file tipo2
    df_tipo_annotator = pd.concat([pd.read_csv(file) for file in tipo2_files])

    return df_tipo_model, df_tipo_annotator


def preprocess_files(df_tipo1, df_tipo2):
    # Pulizia dei dati
    df_tipo1 = df_tipo1.dropna(subset=['comment_id', 'text', 'label', 'annotator_id', 'social_group'])
    df_tipo2 = df_tipo2.dropna(subset=['comment_id', 'text', 'label', 'annotator_id', 'social_group'])

    return df_tipo1, df_tipo2


def calculate_fraction_agreement(df_tipo2):
    # Calcoliamo la percentuale di annotatori che hanno usato la stessa label
    #print(df_tipo2.columns)
    total_annotators = df_tipo2.groupby('comment_id')['annotator_id'].transform('nunique')

    # Calcoliamo il numero di annotatori che hanno usato la stessa label per ciascun 'annotation_id'
    same_label_annotators = df_tipo2.groupby(['comment_id', 'label'])['annotator_id'].transform('nunique')

    # Calcoliamo la percentuale di annotatori che hanno usato la stessa label
    same_label_percentage = (same_label_annotators / total_annotators) * 100

    # Crea un nuovo DataFrame che contiene tutte le colonne originali + la colonna 'same_label_percentage'
    df_with_percentage = df_tipo2.copy()  # Creiamo una copia del DataFrame originale

    df_with_percentage['fraction_agreement'] = same_label_percentage  # Aggiungiamo la nuova colonna


    #print(df_with_percentage)

    df_with_percentage = df_with_percentage[['comment_id', 'annotator_id', 'fraction_agreement']]

    return df_with_percentage


def calculate_brier_scores(df_tipo1, df_tipo2):
    # Uniamo i due dataframe sulla base di comment_id e annotator_id
    df_merged = pd.merge(df_tipo1, df_tipo2, on=['comment_id', 'annotator_id'], suffixes=('_model', ''))
    #print(df_merged.columns)

    # Calcoliamo la media del Brier score per ogni comment_id
    avg_brier_score = df_merged.groupby('comment_id')['brier_score'].mean().reset_index(name='avg_brier_score')

    return df_merged, avg_brier_score


def calculate_quartiles(df_merged):
    # Calcoliamo i quartili Q1, Q2 (mediana) e Q3 per ogni comment_id
    quartiles = df_merged.groupby('comment_id')['brier_score'].quantile([0.25, 0.5, 0.75]).unstack().reset_index()
    quartiles.columns = ['comment_id', 'Q1', 'Q2', 'Q3']
    return quartiles


def aggregate_data(df_tipo_model, df_tipo_annotator, output_folder, modello, dataset,i):
    # Preprocessamento dei file
    df_tipo_model, df_tipo_annotator = preprocess_files(df_tipo_model, df_tipo_annotator)

    # Calcolare frazione di accordo
    fraction_agreement = calculate_fraction_agreement(df_tipo_annotator)

    # Calcolare Brier scores e media
    df_merged, avg_brier_score = calculate_brier_scores(df_tipo_model, df_tipo_annotator)

    # Calcolare quartili
    quartiles = calculate_quartiles(df_merged)

    # Uniamo i dati
    #print("df_merged.columns")
    #print(df_merged.columns)
    #print("fraction_agreement.columns")
    #print(fraction_agreement.columns)
    #print("avg_brier_score.columns")
    #print(avg_brier_score.columns)
    #print("quartiles.columns")
    #print(quartiles.columns)
    final_df = pd.merge(df_merged, fraction_agreement,  on=['comment_id', 'annotator_id'])
    final_df = pd.merge(final_df, avg_brier_score, on='comment_id')
    final_df = pd.merge(final_df, quartiles, on='comment_id')

    # Funzione che somma i valori di 'colonna_1' e 'colonna_2'
    def brier(probs):
        probs=json.loads(probs)
        label=max(probs, key=probs.get)
        conf_scores = dict()
        for pred, prob in probs.items():
            if int(pred) == label:
                conf_score = (1 - prob) ** 2
                conf_scores[pred] = conf_score
            else:
                conf_score = (0 - prob) ** 2
                conf_scores[pred] = conf_score
        non_conformity = np.mean([x for x in conf_scores.values()])

        return non_conformity


    # Applicare la funzione per creare una nuova colonna 'colonna_3'
    final_df['brier_score_model'] = final_df['probs'].apply(brier)

    map_social_group={'Male':0, 'Female':1,
    'Man Non-Western':0, 'Woman Non-Western':1, 'Other Non-Western':-1, 'Woman Western':1,
     'Man Western':0, 'Other Western':-1,'Prefer not to say':-1,
    'man black':0, 'man white':0, 'woman white':1, 'woman black':1, 'man hisp':0, 'na na':-1,
     'man other':0, 'man native':0, 'nonBinary white':-1, 'woman middleEastern':1, 'man na':0, 1:0, 4:1, 2:0, 3:1}

    final_df['social_group']=final_df['social_group'].apply(lambda x: map_social_group[x])

    #print(final_df.columns)
    # Selezioniamo le colonne richieste
    final_df = final_df[['comment_id', 'text_model', 'annotator_id', 'social_group', 'label', 'fraction_agreement',
                         'label_model', 'probs', 'brier_score', 'avg_brier_score', 'Q1', 'Q2', 'Q3','brier_score_model']]

    # Numero di annotatori
    num_annotators = df_merged.groupby('comment_id')['annotator_id'].nunique().reset_index(name='num_annotators')
    final_df = pd.merge(final_df, num_annotators, on='comment_id')

    ######
    #
    # inizio aggunta randomica
    #
    #####
    label_probs = final_df[final_df['social_group'] == 0]['label'].value_counts(normalize=True)
    print(label_probs)
    # MAN
    labels = label_probs.index
    probs = label_probs.values

    final_df['label'] = np.random.choice(
        labels,
        size=len(final_df),
        p=probs
    )

    def brier_label(probs,label):
        probs=json.loads(probs)
        conf_scores = dict()
        for pred, prob in probs.items():
            if int(pred) == label:
                conf_score = (1 - prob) ** 2
                conf_scores[pred] = conf_score
            else:
                conf_score = (0 - prob) ** 2
                conf_scores[pred] = conf_score
        non_conformity = np.mean([x for x in conf_scores.values()])

        return non_conformity

    final_df['brier_score'] = final_df.apply(
        lambda row: brier_label(row['probs'], row['label']),
        axis=1
    )


    # Salviamo il file di output
    output_file = os.path.join(output_folder, f"MANstep_1_{modello}_{dataset}_{i}.csv")
    final_df.to_csv(output_file, index=False)

    print(f"File {output_file} generato con successo.")


# Funzione principale per eseguire il processo per tutti i modelli e dataset
def process_all_data(output_folder,i):
    # Troviamo tutti i dataset disponibili nella cartella
    dataset_pattern = os.path.join("data/measuring_hatespeech", "* - *.csv")
    all_files = glob.glob(dataset_pattern)
    # Estraiamo i dataset unici
    datasets = set()
    for file in all_files:
        filename = os.path.basename(file)
        dataset_name = filename.split(' - ')[0]
        #print(dataset_name)
        datasets.add(dataset_name)
    #print(datasets)

    # Eseguiamo il processo per ogni dataset e modello
    for dataset in datasets:
        # Troviamo i modelli associati a ogni dataset
        modello_pattern = os.path.join(output_folder, f"results_*_{dataset}.csv")
        modello_files = glob.glob(modello_pattern)
        # Estraiamo i modelli unici
        modelli = set()
        print(output_folder)
        for modello_file in modello_files:
            modello_name = os.path.basename(modello_file).split('_')[1]
            modelli.add(modello_name)

        # Eseguiamo il processo per ogni modello
        for modello in modelli:
            print(f"Processando modello {modello} e dataset {dataset}...")
            df_tipo_model, df_tipo_annotator = load_files(output_folder, modello, dataset)
            aggregate_data(df_tipo_model, df_tipo_annotator, output_folder, modello, dataset,i)


# Eseguiamo il processo per tutti i dataset e modelli nella cartella di output
output_folder = 'output_def_rand'  # Cambia con il percorso della tua cartella di output

for i in range(0,10):
    process_all_data(output_folder,i)
