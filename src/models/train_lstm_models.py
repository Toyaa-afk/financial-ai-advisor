import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # pour normaliser les données entre 0 et 1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping  # pour arrêter l'entraînement si la perte ne diminue plus
import joblib  # pour sauvegarder les objets Python comme le scaler

# Définitions des chemins vers les dossiers
DATA_DIR = './stock market data files/Cleaned Data Files/'  # dossier contenant les fichiers CSV des actions
MODELS_DIR = './Models/'  # dossier pour sauvegarder les modèles entraînés

# Crée le dossier MODELS si il n'existe pas déjà
os.makedirs(MODELS_DIR, exist_ok=True)

# Paramètres du modèle et de l'entraînement
SEQ_LEN = 60  # nombre de jours utilisés pour prédire le jour suivant
EPOCHS = 50  # nombre de passes sur les données
BATCH_SIZE = 32  # taille des mini-batchs pour l'entraînement

# Fonction pour créer des séquences à partir des prix
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])  # séquence de SEQ_LEN jours
        y.append(data[i+seq_len])    # prix du jour suivant
    return np.array(X), np.array(y)

# Fonction pour construire le modèle LSTM
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))  # première couche LSTM avec 64 unités
    model.add(Dropout(0.2))  # dropout pour éviter l'overfitting
    model.add(LSTM(32))      # deuxième couche LSTM avec 32 unités
    model.add(Dropout(0.2))
    model.add(Dense(1))      # couche de sortie (prix prédits)
    model.compile(optimizer='adam', loss='mse')  # on utilise Adam et MSE comme fonction de perte
    return model

# Boucle sur tous les fichiers CSV dans le dossier des données
for filename in os.listdir(DATA_DIR):
    if not filename.endswith('.csv'):  # ignore les fichiers qui ne sont pas CSV
        continue

    stock_name = filename[:-4]  # nom de l'action (nom du fichier sans .csv)
    print(f'Training model for {stock_name}...')

    # Chargement des données
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df = df.sort_values('Date')  # tri par date croissante
    close_prices = df['Price'].values.reshape(-1, 1)  # on ne prend que les prix de clôture

    # Vérifie que les données sont suffisantes pour créer des séquences
    if len(close_prices) <= SEQ_LEN:
        print(f"Data too short for {stock_name} (length={len(close_prices)}), skipping...")
        continue

    # Normalisation des données entre 0 et 1
    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close_prices)

    # Création des séquences X (entrée) et y (sortie)
    X, y = create_sequences(close_scaled, SEQ_LEN)
    print(f'X shape before reshape: {X.shape}, y shape: {y.shape}')

    if X.size == 0:  # si pas assez de données pour créer des séquences
        print(f"Not enough data to create sequences for {stock_name}, skipping.")
        continue

    # Reshape pour LSTM : [échantillons, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Construction et entraînement du modèle
    model = build_model((SEQ_LEN, 1))
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)  # stop si pas d'amélioration

    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, callbacks=[early_stop])

    # Sauvegarde du modèle et du scaler
    model.save(os.path.join(MODELS_DIR, f'{stock_name}_lstm_model.keras'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f'{stock_name}_scaler.pkl'))

    print(f'Model and scaler saved for {stock_name}.')

print('All models trained and saved.')
