import pandas as pd
import glob
import os

# Cartella contenente i file CSV di input
input_folder = "../output_def"
output_folder = "../output_def"  # puoi cambiare cartella di destinazione se vuoi

# Pattern dei file di input
file_pattern = os.path.join(input_folder, "step_1_*.csv")

# Itera su tutti i file che rispettano il pattern
for file_path in glob.glob(file_pattern):
    # Leggi il CSV
    df = pd.read_csv(file_path)
    print(file_path)

    # Raggruppa per annotator_id
    grouped = df.groupby('annotator_id')


    # Calcola le statistiche richieste
    summary = grouped.agg(
        brier_score_avg=('brier_score', 'mean'),
        brier_score_Q1=('brier_score', lambda x: x.quantile(0.25)),
        brier_score_Q2=('brier_score', lambda x: x.quantile(0.50)),
        brier_score_Q3=('brier_score', lambda x: x.quantile(0.75)),
        fraction_agreement_avg=('fraction_agreement', 'mean'),
        fraction_agreement_Q1=('fraction_agreement', lambda x: x.quantile(0.25)),
        fraction_agreement_Q2=('fraction_agreement', lambda x: x.quantile(0.50)),
        fraction_agreement_Q3=('fraction_agreement', lambda x: x.quantile(0.75)),
        num_annotation=('annotator_id', 'count'),
        social_group=('social_group', 'first')  # prende il primo social_group incontrato
    ).reset_index()

    # Crea il nome del file di output
    base_name = os.path.basename(file_path)
    output_name = base_name.replace("step_1_", "step_2_")
    output_path = os.path.join(output_folder, output_name)

    # Salva il nuovo CSV
    summary.to_csv(output_path, index=False)

print("Elaborazione completata!")