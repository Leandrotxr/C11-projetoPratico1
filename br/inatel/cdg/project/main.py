import pandas as pd
import numpy as np

dataset = pd.read_csv('../data/student_lifestyle_dataset.csv', delimiter=';')

print("✅ Dataset carregado com sucesso!\n")

dataset = dataset.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
dataset = dataset.replace(r'^\s*$', np.nan, regex=True)


print(dataset)
