import pandas as pd
import numpy as np

# Carica i file CSV
raters_df = pd.read_csv('d3-raters.csv')
ratings_df = pd.read_csv('d3-ratings.csv')
items_df = pd.read_csv('d3-items.csv')


# Unisci i dati di rating con quelli dei raters usando il 'rater_id'
rating_rater_df = pd.merge(ratings_df, raters_df[['rater_id', 'Gender', 'Region', 'Country']], on='rater_id', how='left')

# Unisci i dati ottenuti con quelli degli item usando 'item_id'
combined_df = pd.merge(rating_rater_df, items_df[['item_id', 'text']], on='item_id', how='left')


# Seleziona solo le colonne necessarie per il risultato finale
final_df = combined_df[['item_id', 'text', 'rater_id', 'rating_raw', 'Gender', 'Region', 'Country']]
final_df = final_df[final_df['rating_raw'] >= 0].copy()  # Creare una copia esplicita
final_df.loc[:, 'Western'] = 'Non-Western'  # Imposta un valore di default
final_df.loc[final_df['Region'].isin(['Western Europe', 'North America']), 'Western'] = 'Western'

final_df['social_group'] = final_df['Gender'].str.cat(final_df['Western'], sep=' ')
final_df = final_df.rename(columns={
    'item_id': 'comment_id',
    'rater_id': 'annotator_id',
    'rating_raw': 'label',
})
# Salva il risultato in un nuovo CSV
final_df.to_csv('../davani - offensiveness.csv', index=False)

print("File 'combination.csv' creato con successo.")
