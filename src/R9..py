import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import pairwise_distances

# Cartella contenente i file CSV
cartella_output = 'data/measuring_hatespeech'

# Lista dei file CSV nella cartella di output
file_csvs = [f for f in os.listdir(cartella_output) if ' - ' in f and f.endswith('.csv')]

# Dizionari per memorizzare i dati
right_values = {}
p_values = {}


# Funzione per esplorare il dataframe
def explore_dataframe(df):
    print("Prime righe del dataframe:")
    print(df.head())
    print("\nDescrizione del dataframe:")
    print(df.describe())
    print("\nDistribuzione dei gruppi sociali (0 = uomo, 1 = donna):")
    print(df['social_group'].value_counts())

    # Distribuzione delle etichette per social_group
    print("\nDistribuzione delle annotazioni per ogni gruppo sociale (0 = uomo, 1 = donna):")
    annotazioni_per_gruppo = df.groupby(['social_group', 'label']).size().unstack(fill_value=0)
    print(annotazioni_per_gruppo)


# Itera su ogni file CSV
for file_csv in file_csvs:
    print("\nProcessing file:", file_csv)

    # Carica il dataframe
    df = pd.read_csv(os.path.join(cartella_output, file_csv))

    # Ottieni il nome del modello e del dataset
    dataset_name = file_csv.split('-')[-1].split(".")[0]
    if 'annotatorGender' in df.columns:
        df['social_group']=df['annotatorGender']
    if 'Gender' in df.columns:
        df['social_group']=df['Gender']
    df = df.dropna(subset=['comment_id', 'label', 'annotator_id', 'social_group'])
    df = df.dropna(subset=['comment_id',  'label', 'annotator_id', 'social_group'])

    print(f"Dataset: {dataset_name}")
    map_social_group={'Male':0, 'Female':1,'man':0, 'woman':1,'Man':0, 'Woman':1,
    'Man Non-Western':0, 'Woman Non-Western':1, 'Other Non-Western':-1, 'Woman Western':1,
     'Man Western':0, 'Other Western':-1,
    'man black':0, 'man white':0, 'woman white':1, 'woman black':1, 'man hisp':0, 'na na':-1,
     'man other':0, 'man native':0, 'nonBinary white':-1, 'woman middleEastern':1, 'man na':0, 1:0, 4:1, 2:0, 3:1}
    df=df[df['social_group'].isin(map_social_group.keys())]
    df['social_group']=df['social_group'].apply(lambda x: map_social_group[x])

    # Filtra i dati per includere solo uomini (0) e donne (1)
    df_filtrato = df[df['social_group'].isin([0, 1])]

    # Esplora il dataframe
    explore_dataframe(df_filtrato)

    # Calcola la media delle annotazioni per uomini e donne
    mean_men = df_filtrato[df_filtrato['social_group'] == 0]['label'].mean()
    mean_women = df_filtrato[df_filtrato['social_group'] == 1]['label'].mean()

    print(f"\nMedia delle annotazioni (uomini): {mean_men:.2f}")
    print(f"Media delle annotazioni (donne): {mean_women:.2f}")

    # Confronto delle annotazioni: Test statistico
    # Test t di Student se i dati sono normali, altrimenti U test di Mann-Whitney
    men_ratings = df_filtrato[df_filtrato['social_group'] == 0]['label']
    women_ratings = df_filtrato[df_filtrato['social_group'] == 1]['label']

    # Verifica della normalità con il test di Shapiro
    # (In alternativa, puoi usare un altro test di normalità, ma qui ti mostro una verifica di base)
    from scipy.stats import shapiro

    stat_men, p_men = shapiro(men_ratings)
    stat_women, p_women = shapiro(women_ratings)

    print(f"\nTest di normalità per uomini (Shapiro-Wilk): Stat={stat_men:.3f}, p={p_men:.3f}")
    print(f"Test di normalità per donne (Shapiro-Wilk): Stat={stat_women:.3f}, p={p_women:.3f}")

    # Se i dati non sono normali, usa il test di Mann-Whitney U
    if p_men < 0.05 or p_women < 0.05:
        print("\nTest non parametrici (Mann-Whitney U test):")
        stat, p = mannwhitneyu(men_ratings, women_ratings)
        print(f"Statistiche: {stat:.3f}, p-value: {p:.3f}")
    else:
        print("\nTest parametrici (T-test):")
        stat, p = ttest_ind(men_ratings, women_ratings)
        print(f"Statistiche: {stat:.3f}, p-value: {p:.3f}")

    # Salva il p-value per il test statistico
    p_values[dataset_name] = p

    # Visualizza la distribuzione delle annotazioni per uomini e donne
    plt.figure(figsize=(8, 6))
    sns.histplot(men_ratings, kde=True, color='blue', label='Uomini', bins=5, alpha=0.6)
    sns.histplot(women_ratings, kde=True, color='pink', label='Donne', bins=5, alpha=0.6)
    plt.legend()
    plt.title(f"Distribuzione delle annotazioni: {dataset_name}")
    plt.xlabel("Rating")
    plt.ylabel("Frequenza")
    plt.show()

    # Calcola e salva il punteggio medio per il gruppo
    right_values[dataset_name] = {
        'men_mean': mean_men,
        'women_mean': mean_women,
        'p_value': p
    }

# Visualizza i risultati finali
print("\nP-values dei test di differenza tra uomini e donne:")
for dataset, values in right_values.items():
    print(f"{dataset}: {values['p_value']:.3f}")
