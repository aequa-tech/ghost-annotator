import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from R_utilities import mappatura_dataset_colore
from src.R_utilities import mappatura_dataset
from matplotlib.ticker import FuncFormatter

# Lista dei file CSV da caricare
files = ['data/measuring_hatespeech/attitudes - hate speech.csv',
         'data/measuring_hatespeech/cade - acceptability.csv',
         'data/measuring_hatespeech/corpus - violence.csv',
         'data/measuring_hatespeech/davani - offensiveness.csv']  # Sostituisci con i tuoi file

# Crea il grafico
plt.figure(figsize=(10, 6))

# Per ogni file, carica i dati, calcola la cumulativa e disegna la linea
for file in files:
    # Carica i dati dal CSV
    dataset_name = file.split('/')[-1].split('-')[0].strip()

    df_annotators = pd.read_csv(file)
    df_annotators = df_annotators.dropna(subset=['comment_id', 'annotator_id'])
    df_annotators = df_annotators[['comment_id', 'annotator_id']]

    # Calcola la cumulativa del numero di annotatori per ogni commento
    df_annotators = df_annotators.groupby('comment_id')['annotator_id'].nunique().reset_index()

    # Rinomina la colonna per il numero di annotatori
    df_annotators.rename(columns={'annotator_id': 'num_annotatori'}, inplace=True)
    print(dataset_name,df_annotators["num_annotatori"].min(),df_annotators["num_annotatori"].max())
    # Traccia la linea per questo file
    sns.ecdfplot(df_annotators['num_annotatori'], color=mappatura_dataset_colore[dataset_name], label=mappatura_dataset[dataset_name], stat="percent", zorder=3, linewidth=2)

# Personalizza l'asse X per includere >100
plt.xticks(list(range(0, 81, 10)) + [81], labels=list(range(0, 81, 10)) + ['>=100'])
plt.xlim(0, 80)
plt.ylim(-0.1, 110)
# Personalizza il grafico
plt.title('Cumulative Distribution of the Number of Annotators per Comment', fontweight='bold')
plt.xlabel('Number of Annotators per Comment')
plt.ylabel('Cumulative Frequency')
plt.legend(title='Datasets', loc='lower right')
plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)



# Mostra il grafico
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
