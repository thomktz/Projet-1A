#############################
### OLD FILE, USELESS NOW ###
#############################
import pandas as pd

sequence = ["a", "sigma", "forall", "thereex", "integral", "real", "in", "comma","a", "sigma", "forall", "thereex", "integral", "real", "in", "comma"]
number = 48

out = []

for n in range(len(sequence)):
    for i in range(number):
        out.append(n%8)
import numpy as np

np.save("labels_projet.npy", out)

data = np.load("labels_projet.npy")

print(data)
