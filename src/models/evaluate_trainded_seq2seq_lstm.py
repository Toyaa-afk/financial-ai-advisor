import os
import pandas as pd
import numpy as np
import joblib  # pour charger les scalers sauvegardés
from tensorflow.keras.models import load_model  # pour charger les modèles LSTM Seq2Seq

# ---- Paramètres ----
DATA_DIR = "./stock market data files/Cleaned Data Files/"  # dossier des fichiers CSV
MODELS_BASE_DIR = "./Models/"  # dossier contenant les modèles pour chaque horizon
SEQ_LEN = 60  # longueur de la séquence d'entrée
HORIZONS = ["1month", "3months", "6months", "9months", "1year"]  # horizons de prédiction

# Dossier pour sauvegarder les résultats d'évaluation
EVAL_DIR = os.path.join(MODELS_BASE_DIR, "seq2seq_evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

# Fichier pour log des erreurs
ERROR_LOG = os.path.join(EVAL_DIR, "errors.log")

# ---- Fonction: enregistrer les erreurs ----
def log_error(msg):
    """
    Enregistre les erreurs dans un fichier et affiche à l'écran
    """
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print("[ERROR]", msg)

# ---- Fonction: évaluer un modèle pour une action et un horizon donné ----
def evaluate_model(stock_name, horizon):
    """
    Charge le modèle Seq2Seq et le scaler, prépare la dernière séquence,
    prédit les prix pour l'horizon, et sauvegarde les résultats dans un CSV.
    """
    model_path = f"{MODELS_BASE_DIR}{horizon}/{stock_name}_lstm_model_{horizon}.keras"
    scaler_path = f"{MODELS_BASE_DIR}{horizon}/{stock_name}_scaler_{horizon}.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        log_error(f"Missing model or scaler for {stock_name} - {horizon}")
        return

    try:
        # Chargement du modèle et du scaler
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)

        # Chargement des prix
        df = pd.read_csv(f"{DATA_DIR}{stock_name}.csv")
        close_prices = df["Price"].values.reshape(-1, 1)

        # Préparer la dernière séquence pour la prédiction
        last_seq = close_prices[-SEQ_LEN:]
        last_seq_scaled = scaler.transform(last_seq)
        X_input = np.expand_dims(last_seq_scaled, axis=0)  # reshape pour le LSTM

        # Prédiction
        prediction_scaled = model.predict(X_input)
        prediction_scaled = np.array(prediction_scaled).reshape(-1, 1)

        # Gestion des valeurs invalides éventuelles
        prediction_scaled = np.nan_to_num(prediction_scaled, nan=0.0, posinf=1e6, neginf=-1e6)

        # Revenir à l'échelle originale
        prediction = scaler.inverse_transform(prediction_scaled)

        # Sauvegarde du résultat dans un CSV
        horizon_eval_dir = os.path.join(EVAL_DIR, horizon)
        os.makedirs(horizon_eval_dir, exist_ok=True)

        output_path = os.path.join(horizon_eval_dir, f"{stock_name}.csv")
        pd.DataFrame(prediction, columns=["Predicted Price"]).to_csv(output_path, index=False)

        print(f"[OK] {stock_name} - {horizon} saved to {output_path}")

    except Exception as e:
        log_error(f"Error evaluating {stock_name} - {horizon}: {str(e)}")

# ---- Exécution principale ----
if __name__ == "__main__":
    # Liste de toutes les actions à évaluer
    all_stocks = [f.replace(".csv", "") for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

    # Évaluer toutes les actions pour tous les horizons
    for stock in all_stocks:
        for horizon in HORIZONS:
            evaluate_model(stock, horizon)

    print("\nEvaluation completed. Check results in:", EVAL_DIR)
    print("Errors logged in:", ERROR_LOG)
