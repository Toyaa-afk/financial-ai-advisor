import pandas as pd
import numpy as np
import os

# ---- Fonction: nettoyer un fichier CSV de données boursières ----
def clean_stock_data(file_path):
    """
    Nettoie les données d'une action :
    - Conversion des dates en format datetime
    - Transformation des volumes (K -> milliers, M -> millions)
    - Transformation des pourcentages de changement en float
    - Remplissage des jours manquants (jours ouvrés)
    - Interpolation des prix manquants
    """
    try:
        df = pd.read_csv(file_path)

        # Conversion de la colonne Date au format datetime (mois/jour/année)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')

        # Tri par date
        df = df.sort_values('Date').reset_index(drop=True)

        # Fonction pour convertir les volumes K/M en valeurs numériques
        def convert_volume(vol_str):
            if pd.isna(vol_str) or vol_str == '':
                return np.nan
            vol_str = str(vol_str).strip().upper()
            if vol_str.endswith('K'):
                return float(vol_str[:-1]) * 1_000
            elif vol_str.endswith('M'):
                return float(vol_str[:-1]) * 1_000_000
            else:
                try:
                    return float(vol_str)
                except:
                    return np.nan

        df['Vol.'] = df['Vol.'].apply(convert_volume)

        # Convertir le pourcentage de changement en float (0-1)
        df['Change %'] = df['Change %'].str.rstrip('%').astype(float) / 100.0

        # Créer toutes les dates ouvrées entre la première et la dernière date
        all_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='B')
        full_df = pd.DataFrame({'Date': all_dates})

        # Merge pour inclure tous les jours ouvrés
        df = pd.merge(full_df, df, on='Date', how='left')

        # Interpolation linéaire pour les prix manquants
        df[['Price', 'Open', 'High', 'Low']] = df[['Price', 'Open', 'High', 'Low']].interpolate(method='linear')

        # Remplissage des volumes et changements manquants
        df['Vol.'] = df['Vol.'].fillna(0)
        df['Change %'] = df['Change %'].fillna(0)

        # Vérification des gaps encore présents
        missing_after_fill = df['Price'].isna().sum()
        if missing_after_fill / len(df) > 0.05:  # plus de 5% de données manquantes
            print(f"Warning: Large gaps remain in {os.path.basename(file_path)} - skipping")
            return None

        return df

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ---- Fonction: nettoyer un dossier entier de fichiers CSV ----
def batch_clean_folder(folder_path, save_cleaned=False, cleaned_folder=None):
    """
    Parcourt tous les CSV dans un dossier, les nettoie et les retourne dans un dictionnaire.
    Optionnellement, sauvegarde les fichiers nettoyés dans un dossier.
    """
    cleaned_data = {}
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if save_cleaned and cleaned_folder:
        os.makedirs(cleaned_folder, exist_ok=True)

    for file in files:
        path = os.path.join(folder_path, file)
        print(f"Processing {file} ...")
        cleaned_df = clean_stock_data(path)

        if cleaned_df is not None:
            cleaned_data[file] = cleaned_df
            if save_cleaned and cleaned_folder:
                cleaned_path = os.path.join(cleaned_folder, file)
                cleaned_df.to_csv(cleaned_path, index=False)
        else:
            print(f"Skipping {file} due to insufficient data.")

    return cleaned_data

# ---- Exemple d'utilisation ----
folder_path = r"C:\AI advisor\stock market data files\Raw Data Files"
cleaned_folder = r"C:\AI advisor\stock market data files\Cleaned Data Files"

# Pour nettoyer et sauvegarder les fichiers nettoyés :
cleaned_dfs = batch_clean_folder(folder_path, save_cleaned=True, cleaned_folder=cleaned_folder)

# Pour garder les données nettoyées en mémoire uniquement :
# cleaned_dfs = batch_clean_folder(folder_path)
