import pandas as pd
import numpy as np

def calcul_metrics(ch, invest_per):
    df = pd.read_csv(ch)
    # Trier par date décroissante
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=False).reset_index(drop=True)
    
    # Sélectionner les données en fonction de la période d'investissement
    months_seen = 0
    prev_month = df.loc[0, 'Date'].month
    selected_rows = []
    
    for i, date in enumerate(df['Date']):
        current_month = date.month
        if current_month != prev_month:
            months_seen += 1
            prev_month = current_month
        if months_seen >= invest_per:
            break
        selected_rows.append(i)
    
    df_period = df.loc[selected_rows]
    
    # Convertir 'Change %' colonne en float
    df_period['Daily Return %'] = (df_period['Change %'].str.replace('%', '').astype(float) )/ 100
    
    average_daily_return = ((np.log1p(df_period['Daily Return %'])).mean()) 
    std_daily_return = (df_period['Daily Return %'].std()) 
    
    # Annualiser le rendement et le risque pour plus de cohérence
    annual_return = np.exp(average_daily_return * 250) - 1
    annual_risk = std_daily_return * np.sqrt(250)
    
    return [float(f"{average_daily_return*100:.4f}"), 
            float(f"{std_daily_return*100:.4f}"), 
            float(f"{annual_return*100:.2f}"), 
            float(f"{annual_risk*100:.2f}")]