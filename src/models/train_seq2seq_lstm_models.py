import os
import glob  # pour lister tous les fichiers CSV dans un dossier
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # pour normaliser les prix entre 0 et 1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping  # pour arrêter l'entraînement si la perte stagne
import joblib  # pour sauvegarder le scaler

# ---- Paramètres ----
DATA_DIR = "./stock market data files/Cleaned Data Files/"  # dossier contenant les fichiers CSV
MODELS_DIR = "./Models/"  # dossier pour sauvegarder les modèles
SEQ_LEN = 60  # nombre de jours passés utilisés comme entrée

# Horizons de prédiction (nombre de jours dans le futur)
HORIZONS = {
    "1month": 30,
    "3months": 90,
    "6months": 180,
    "9months": 270,
    "1year": 365
}

# ---- Création des sous-dossiers pour chaque horizon ----
for horizon in HORIZONS.keys():
    os.makedirs(os.path.join(MODELS_DIR, horizon), exist_ok=True)

# ---- Fonction: Préparer les séquences pour l'entraînement ----
def prepare_sequences(data, seq_len, n_ahead):
    """
    Transforme les prix en séquences X (entrée) et y (sortie)
    X : séquence de seq_len jours
    y : séquence des n_ahead jours suivants
    """
    X, y = [], []
    for i in range(len(data) - seq_len - n_ahead):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+n_ahead])
    return np.array(X), np.array(y)

# ---- Fonction: Construire un modèle Seq2Seq LSTM ----
def build_seq2seq_lstm(seq_len, n_ahead):
    """
    Modèle sequence-to-sequence :
    - Encoder : résume la séquence d'entrée en un vecteur
    - RepeatVector : répète le vecteur pour correspondre à la longueur de sortie
    - Decoder : génère la séquence prédite
    """
    # Encoder
    encoder_inputs = Input(shape=(seq_len, 1))
    encoder_lstm = LSTM(100, activation='relu')(encoder_inputs)
    encoder_output = RepeatVector(n_ahead)(encoder_lstm)
    
    # Decoder
    decoder_lstm = LSTM(100, activation='relu', return_sequences=True)(encoder_output)
    decoder_outputs = TimeDistributed(Dense(1))(decoder_lstm)

    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# ---- Boucle principale pour entraîner tous les modèles ----
csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))  # récupère tous les CSV du dossier

for csv_path in csv_files:
    stock_name = os.path.splitext(os.path.basename(csv_path))[0]  # nom de l'action
    print(f"\n Training models for {stock_name}...")

    # Chargement et normalisation des données
    df = pd.read_csv(csv_path)
    if "Price" not in df.columns:
        raise ValueError(f"CSV {stock_name} does not have 'Price' column.")

    prices = df["Price"].values.reshape(-1, 1)  # on prend uniquement les prix
    scaler = MinMaxScaler(feature_range=(0, 1))  # normalisation entre 0 et 1
    scaled_prices = scaler.fit_transform(prices)

    # Entraînement pour chaque horizon
    for horizon_name, days_ahead in HORIZONS.items():
        print(f"  ➡ Horizon: {horizon_name} ({days_ahead} days ahead)")

        # Préparer les séquences X et y
        X, y = prepare_sequences(scaled_prices, SEQ_LEN, days_ahead)

        # Construire le modèle Seq2Seq
        model = build_seq2seq_lstm(SEQ_LEN, days_ahead)

        # Entraînement avec EarlyStopping pour éviter l'overfitting
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=50, batch_size=32, verbose=0, callbacks=[es])

        # Sauvegarde du modèle
        model_save_path = os.path.join(MODELS_DIR, horizon_name, f"{stock_name}_lstm_model_{horizon_name}.keras")
        model.save(model_save_path)

        # Sauvegarde du scaler correspondant
        scaler_save_path = os.path.join(MODELS_DIR, horizon_name, f"{stock_name}_scaler_{horizon_name}.pkl")
        joblib.dump(scaler, scaler_save_path)

        print(f"    Saved: {model_save_path}")
        print(f"    Saved: {scaler_save_path}")

print("\n Training completed for all stocks and horizons.")
