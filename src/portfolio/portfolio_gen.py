import pandas as pd
import numpy as np
import csv
from metrics import calcul_metrics
from csv_file_crator import csv_cre
from selector import asset_select
from weight_generator import*

def portfolios(bud,ris,per,act):
    #importation des fichier necessaires 
    prices = pd.read_csv("./portfolio_optimiser_data_files/Tunisia Stocks Prices.csv")
    stats = pd.read_csv("./portfolio_optimiser_data_files/Tunisia Stocks Stats.csv")
    
    #fusion des deux dictionnaires
    comb = pd.merge(prices, stats, on="Name")
    # --- creation d'un 3eme csv ---
    #nom des colonne 
    fields = ['Name','Avg_Da_Ret%','Da_Std_Dev%','Exp_ann_Ret%','Ann_Std_Dev%']

    rows=[]#liste des lignes 
    for name in comb['Name']:
        l=[]
        #nom du fichier correspendant a l'historique de chaque societe(exm:ATB Stock Price History.csv")
        f_name=f"./portfolio_optimiser_data_files/{name} Stock Price History.csv"
        l.append(name)
        #calcul_metrics retourne une liste de statistique d'un tel societe
        l.extend(calcul_metrics(f_name,per))
        rows.append(l)

    filename = "./portfolio_optimiser_data_files/market_metrics.csv"
    csv_cre(filename,fields,rows)#creation du fich csv
    # --- creation du fichier full stats ---
    metrics_csv = pd.read_csv(filename)
    final = pd.merge(comb, metrics_csv, on='Name')
    final.to_csv("./portfolio_optimiser_data_files/Tunisia Stocks Full Dataset.csv", index=False)
    
    
    # --- creation d'une liste de portfolio ---
    #selection les meilleurs actifs selon un nombre donné
    top_assets=asset_select("./portfolio_optimiser_data_files/Tunisia Stocks Full Dataset.csv",bud,ris,per,act)
    assets=top_assets["Name"].tolist()
    
    # generer un nombre de poid
    matrix=[]
    weights=weight_gen(len(top_assets))
    
    #calcul de chaque portfolio par rapport au poid
    for items in weights:
        #variables pour calcul
        Exp_Ret_per=0 #taux de rendement attendu
        Exp_std_dev=0 #écart type attendu
        for i in range (len(items)):
            Exp_Ret_per=Exp_Ret_per+(items[i]/100)*top_assets.iloc[i]["Exp_ann_Ret%"]
            Exp_std_dev=Exp_std_dev+((items[i]/100)**2)*(top_assets.iloc[i]["Ann_Std_Dev%"]**2)
        risk_free_rate=5 #avg en tunis d'apres chat
        sharp_ratio=(Exp_Ret_per-risk_free_rate)/Exp_std_dev
        matrix.append([Exp_Ret_per,np.sqrt(Exp_std_dev),sharp_ratio,risk_free_rate,items,assets])
    return(matrix)