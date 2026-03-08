import pandas as pd
import numpy as np

def optim_actif(weights,budget):
    #intialisation des var
    df = pd.read_csv("./portfolio_optimiser_data_files/Tunisia Stocks Prices.csv")
    shares = {}
    percentage={}
    leftover = budget
    #division de chaque pourcentage et calcul de rest total
    for name, weight in weights.items():
        price = df.loc[df["Name"] == name, "Last"].values[0]
        amount = budget * (weight/100)
        num_shares = int(amount // price)
        perc=((num_shares*price)*100)/budget
        perc=round(perc, 2)
        spent = num_shares * price
        #pourcentages optimisées
        shares[name] = num_shares
        percentage[name]=f"{perc}%"
        leftover -= spent

    print("les pourcentage a investir dans chaque actifs:", percentage)
    print("nombres d'actifs a investir:", shares)
    print("Argent restant:", round(leftover, 2))
