import numpy as np
#generer une liste de n pourcentages vaut a 100 (on appelle poids)
def generate_random_percentages(n, round_to=2):
    random_numbers = np.random.rand(n)
    normalized = random_numbers / random_numbers.sum()
    percentages = normalized * 100
    
    percentages = np.round(percentages, round_to)
    diff = 100 - percentages.sum()
    
    percentages[np.argmax(percentages)] += diff
    return percentages
#generer un liste contient plusieurs poids
def weight_gen(n):
    weights=[]
    for i in range(1000):
        l=[]
        result = generate_random_percentages(n)
        l.extend(result)
        weights.append(l)
    return (weights)

