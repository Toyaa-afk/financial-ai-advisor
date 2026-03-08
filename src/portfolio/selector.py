import pandas as pd

def asset_select(name,budget,risk_level,investment_period,assets_number):
    #lire le fichier qui contient les infos du marché
    df = pd.read_csv(name)

    
    #selectionner les actifs selon leurs risk
    selected = df[df['Ann_Std_Dev%'] <= risk_level]

    #tri des actif par rapport au rendement annuel decroisssant
    selected = selected.sort_values(by='Exp_ann_Ret%',ascending=False)
    
    #selectioner les meiileurs n actifs
    top_stocks = selected.head(assets_number)
    return(top_stocks)

