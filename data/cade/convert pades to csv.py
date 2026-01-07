import pandas as pd


def parquet_to_csv(parquet_path, csv_path):
    # Legge il file Parquet
    df = pd.read_parquet(parquet_path)

    # Salva il DataFrame come file CSV
    df.to_csv(csv_path, index=False)  # index=False per non includere l'indice nel CSV

    print(f"File CSV salvato come {csv_path}")


def main():
    parquet_path = "train-00000-of-00001-c0dc3bc958643d2c.parquet"  # Inserisci il percorso del tuo file Parquet
    csv_path = "corpus_cade.csv"  # Percorso di salvataggio del file CSV

    # Converte il file Parquet in CSV
    parquet_to_csv(parquet_path, csv_path)


if __name__ == "__main__":
    main()
