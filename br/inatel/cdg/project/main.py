import pandas as pd

df = pd.read_csv('../data/data.csv', delimiter=';')

print("✅ Dataset carregado com sucesso!\n")
print(df.head())

