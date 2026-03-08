import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # pour tracer les graphiques prix réel vs prédit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # métriques d'évaluation
import joblib  # pour charger le scaler sauvegardé
from tensorflow.keras.models import load_model  # pour charger le modèle LSTM sauvegardé
from tqdm import tqdm  # pour afficher une barre de progression

# ---- Paramètres ----
DATA_DIR = './stock market data files/Cleaned Data Files/'  # dossier contenant les fichiers CSV
MODELS_DIR = './Models/'  # dossier contenant les modèles entraînés
SEQ_LEN = 60  # longueur des séquences utilisées pour la prédiction
SHOW_PLOTS = False  # mettre à True pour voir les graphiques

# ---- Fonction: créer les séquences pour le test ----
def create_sequences(data, seq_len):
    """
    Transforme les données en séquences X et y pour l'évaluation.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# ---- Fonction: évaluer un modèle pour une action ----
def evaluate_model(stock_name):
    """
    Charge le modèle et le scaler pour une action, prépare les séquences de test,
    fait les prédictions et calcule les métriques MSE, MAE et R².
    """
    model_path = os.path.join(MODELS_DIR, f'{stock_name}_lstm_model.keras')
    scaler_path = os.path.join(MODELS_DIR, f'{stock_name}_scaler.pkl')
    csv_path = os.path.join(DATA_DIR, stock_name + '.csv')

    # Vérifie que le modèle et le scaler existent
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"Skipping {stock_name}: Missing model/scaler.")
        return None

    try:
        # Chargement des données et tri par date
        df = pd.read_csv(csv_path).sort_values('Date')
        close_prices = df['Price'].values.reshape(-1, 1)

        # Vérifie que la longueur des données est suffisante
        if len(close_prices) <= SEQ_LEN:
            print(f"Skipping {stock_name}: Not enough data for sequence length {SEQ_LEN}.")
            return None

        # Chargement du scaler et du modèle
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)

        # Normalisation et préparation des données de test
        close_scaled = scaler.transform(close_prices)
        split_idx = int(len(close_scaled) * 0.8)  # 80% entraînement, 20% test
        test_data = close_scaled[split_idx - SEQ_LEN:]
        X_test, y_test = create_sequences(test_data, SEQ_LEN)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Prédictions
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled)  # retour à l'échelle originale
        y_true = scaler.inverse_transform(y_test)

        # Calcul des métriques d'évaluation
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Affichage du graphique si demandé
        if SHOW_PLOTS:
            plt.figure(figsize=(12,6))
            plt.plot(y_true, label='Actual Price')
            plt.plot(y_pred, label='Predicted Price')
            plt.title(f'{stock_name} - Actual vs Predicted')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            plt.show()

        # Retourne les résultats pour cette action
        return {'Stock': stock_name, 'MSE': mse, 'MAE': mae, 'R2': r2}

    except Exception as e:
        print(f"Error evaluating {stock_name}: {e}")
        return None

# ---- Fonction: évaluer tous les modèles ----
def evaluate_all_stocks():
    results = []
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

    # Boucle sur toutes les actions avec barre de progression
    for csv_file in tqdm(csv_files, desc="Evaluating stocks"):
        stock_name = csv_file[:-4]
        res = evaluate_model(stock_name)
        if res:
            results.append(res)
            # Sauvegarde des résultats après chaque action pour éviter la perte
            pd.DataFrame(results).to_csv('testing_results.csv', index=False)

    print("All results saved to testing_results.csv")

# ---- Exécution principale ----
if __name__ == '__main__':
    evaluate_all_stocks()
