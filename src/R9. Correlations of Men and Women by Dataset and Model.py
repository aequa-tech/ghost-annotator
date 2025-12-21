import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.metrics import pairwise_distances
from scipy.stats import ttest_ind, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
# Cartella contenente i file CSV
cartella_output = 'output'

# Lista dei file CSV nella cartella di output
file_csvs = [f for f in os.listdir(cartella_output) if f.startswith('step_1_') and f.endswith('.csv')]

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

correlations_data=[]
# Itera su ogni file CSV
for file_csv in file_csvs:
    print("\nProcessing file:", file_csv)

    # Carica il dataframe
    df = pd.read_csv(os.path.join(cartella_output, file_csv))
    print(df.columns)
    # Ottieni il nome del modello e del dataset
    model_name = file_csv.split('_')[2]

    dataset_name = file_csv.split('_')[-1].split(".")[0]

    df = df.dropna(subset=['comment_id', 'label', 'annotator_id', 'social_group'])
    df = df.dropna(subset=['comment_id',  'label', 'annotator_id', 'social_group'])

    print(f"Dataset: {dataset_name}")


    # Filtra i dati per includere solo uomini (0) e donne (1)
    df_filtrato = df[df['social_group'].isin([0, 1])]



    # Confronto tra annotazioni umane e annotazioni del modello
    # Creiamo un nuovo dataframe con i dati per gli uomini e le donne separatamente per le annotazioni umane e quelle del modello
    df_filtrato['model_rating'] = df_filtrato['label_model']

    # Uomini
    df_men = df_filtrato[df_filtrato['social_group'] == 0]
    # Donne
    df_women = df_filtrato[df_filtrato['social_group'] == 1]

    # Calcolo delle medie
    mean_men_human = df_men['label'].mean()
    mean_men_model = df_men['model_rating'].mean()
    mean_women_human = df_women['label'].mean()
    mean_women_model = df_women['model_rating'].mean()

    # Stampa delle medie
    print(f"Media delle annotazioni umane (uomini): {mean_men_human:.2f}")
    print(f"Media delle annotazioni del modello (uomini): {mean_men_model:.2f}")
    print(f"Media delle annotazioni umane (donne): {mean_women_human:.2f}")
    print(f"Media delle annotazioni del modello (donne): {mean_women_model:.2f}")

    # Test statistico (t-test o Mann-Whitney) per verificare se ci sono differenze significative
    # per gli uomini
    stat_men, p_men = ttest_ind(df_men['label'], df_men['model_rating'])
    # per le donne
    stat_women, p_women = ttest_ind(df_women['label'], df_women['model_rating'])

    print(f"\nTest t per gli uomini (rating umani vs modello): Stat={stat_men:.3f}, p-value={p_men:.3f}")
    print(f"Test t per le donne (rating umani vs modello): Stat={stat_women:.3f}, p-value={p_women:.3f}")

    # Grafico delle distribuzioni
    #plt.figure(figsize=(10, 6))
    #sns.histplot(df_men['label'], kde=True, color='blue', label='Uomini (Human)', bins=5, alpha=0.6)
    #sns.histplot(df_men['model_rating'], kde=True, color='blue', label='Uomini (Model)', bins=5, linestyle='--', alpha=0.6)
    #sns.histplot(df_women['label'], kde=True, color='pink', label='Donne (Human)', bins=5, alpha=0.6)
    #sns.histplot(df_women['model_rating'], kde=True, color='pink', label='Donne (Model)', bins=5, linestyle='--', alpha=0.6)
    #plt.legend()
    #plt.title("Confronto tra Annotazioni Umane e del Modello")
    #plt.xlabel("Rating")
    #plt.ylabel("Frequenza")
    #plt.show()

    # Calcolo della correlazione tra le annotazioni del modello e quelle degli esseri umani per uomini e donne
    corr_men = np.corrcoef(df_men['label'], df_men['model_rating'])[0, 1]
    corr_women = np.corrcoef(df_women['label'], df_women['model_rating'])[0, 1]

    print(f"\nCorrelazione tra annotazioni umane e del modello per gli uomini: {corr_men:.3f}")
    print(f"Correlazione tra annotazioni umane e del modello per le donne: {corr_women:.3f}")

    # Test di Mann-Whitney U per confrontare le annotazioni del modello tra uomini e donne
    stat_model, p_model = mannwhitneyu(df_men['model_rating'], df_women['model_rating'])

    print(f"\nTest di Mann-Whitney U per il modello (uomini vs donne): Stat={stat_model:.3f}, p-value={p_model:.3f}")
    # Aggiungi i risultati al dataframe
    correlations_data.append({
        'model': model_name,
        'dataset': dataset_name,
        'correlation_men': corr_men,
        'correlation_women': corr_women
    })

# Converti il risultato in un dataframe
df_correlations = pd.DataFrame(correlations_data)
print(df_correlations.columns)
print(df_correlations['dataset'].dtype)
print(df_correlations['dataset'])
# Visualizzazione con un grouped bar chart

# Crea il grafico
plt.figure(figsize=(14, 7))

# Creazione delle barre per correlation_men
ax = sns.barplot(data=df_correlations, x='dataset', y='correlation_men', hue='model', ci=None, dodge=True,alpha=0.6,palette=['#fdae61','#abd9e9','#d7191c','#2c7bb6'])

# Creazione delle barre per correlation_women con un offset (trasparente)
sns.barplot(data=df_correlations, x='dataset', y='correlation_women', hue='model', ci=None, dodge=True, alpha=0.6, palette=['#fdae61','#abd9e9','#d7191c','#2c7bb6'],ax=ax)

# Aggiungi pattern a tratteggio (hatch) per differenziare i generi
for p in ax.patches:
    # Se la barra rappresenta correlation_men, aggiungi un tratteggio verticale
    if p.get_height() in df_correlations['correlation_men'].values:
        p.set_hatch('//')  # Tratteggio diagonale per gli uomini
    # Se la barra rappresenta correlation_women, aggiungi un tratteggio orizzontale
    else:
        p.set_hatch('\\\\')  # Tratteggio diagonale invertito per le donne

# Aggiungi titoli e etichette
plt.title('Correlations of Men and Women by Dataset and Model')
plt.xlabel('Dataset')
plt.ylabel('Correlation')

# Mostra la legenda
plt.legend(title='Model', loc='upper left')


# La legenda dei modelli
handles, labels = plt.gca().get_legend_handles_labels()
# Aggiungi voci per 'Man' e 'Woman' con il tratteggio
legend_handles = handles[:int(len(labels)/2)]  # Rimuove l'ultima voce della legenda (per i modelli)
legend_labels = labels[:int(len(labels)/2)]

# Aggiungi voci per "Man" e "Woman"
man_patch = Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='black', hatch='//', lw=0, label='Man')
woman_patch = Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='black', hatch='\\\\', lw=0, label='Woman')

legend_handles += [man_patch, woman_patch]
legend_labels += ['Man', 'Woman']

# Mostra la legenda personalizzata
plt.legend(handles=legend_handles, labels=legend_labels, title="Model and Gender")


# Ruota le etichette dei dataset per una visibilità migliore
plt.xticks(rotation=45)
plt.tight_layout()

# Mostra il grafico
plt.show()
