import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from R_utilities import mappatura_modelli, mappatura_dataset

# Definisci la cartella contenente i file CSV
cartella_output = '../output_def'  # Sostituisci con il percorso corretto


# Funzione per estrarre i dati necessari per il box plot
def estrai_dati_box_plot(file_csv):
    # Leggi il file CSV
    df = pd.read_csv(file_csv)

    # Raggruppa per comment_id e prendi un valore di brier_score_model unico per ogni comment_id
    df_unique = df[['comment_id', 'brier_score_model']].drop_duplicates()

    # Unisci altre informazioni come il modello e dataset per etichettare correttamente
    model_name = file_csv.split('/')[-1].split('_')[2]
    dataset_name = file_csv.split('_')[-1].split(".")[0]

    df_unique['model_name'] = model_name
    df_unique['dataset_name'] = dataset_name

    return df_unique


# Lista dei file CSV nella cartella di output
file_csvs = [f for f in os.listdir(cartella_output) if f.startswith('step_1_') and f.endswith('.csv')]
print(file_csvs)
# Crea un DataFrame vuoto per contenere i risultati
dati_box_plot = []

# Estrai i dati per ogni file CSV
for file_csv in file_csvs:
    df = estrai_dati_box_plot(os.path.join(cartella_output, file_csv))
    dati_box_plot.append(df)

# Combina tutti i risultati in un unico DataFrame
df_box_plot = pd.concat(dati_box_plot, ignore_index=True)

# Crea un nuovo dataframe con i nomi rimappati per i modelli
df_box_plot['model_name'] = df_box_plot['model_name'].map(mappatura_modelli)
df_box_plot['dataset_name'] = df_box_plot['dataset_name'].map(mappatura_dataset)
print(df_box_plot)
# Creazione del box plot
plt.figure(figsize=(10, 6))

# Ordina i modelli per una visualizzazione chiara
modelli_ordinati = ['Llama-3.2-1B', 'Qwen2.5-1.5B', 'Llama-3.1-8B',
                    'Qwen2.5-7B']  # Sostituisci con i tuoi nomi di modello

# Impostiamo i flierprops per ridurre la dimensione delle palline (outliers)
flierprops = dict(marker='o', markerfacecolor='black', markersize=1, markeredgewidth=0)

sns.boxplot(x='dataset_name', y='brier_score_model', hue='model_name', data=df_box_plot,
            hue_order=modelli_ordinati, palette=['#fdae61','#abd9e9','#d7191c','#2c7bb6'],
            flierprops=flierprops)

# Aggiusta la formattazione
plt.title('Model uncertainty Across Models and Datasets', fontweight='bold')
plt.xlabel('Datasets')
plt.ylabel('Model uncertainty')
plt.xticks(rotation=45)
plt.legend(title="Models")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.grid(axis='y', color='lightgray', linestyle='-', linewidth=0.5, zorder=0)  # Griglia orizzontale, grigia chiara

plt.tight_layout()

# Mostra il grafico
plt.show()
