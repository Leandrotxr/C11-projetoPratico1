import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../data/student_lifestyle_dataset.csv', delimiter=',')

print("Dataset carregado com sucesso!\n")

dataset['Stress_Level'].str.strip()
dataset = dataset.replace(r'^\s*$', np.nan, regex=True)

print(dataset)