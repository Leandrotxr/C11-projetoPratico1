import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/student_lifestyle_dataset.csv', delimiter=',')

print("Dataset carregado com sucesso!\n")

df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
df = df.replace(r'^\s*$', np.nan, regex=True)

#===================================================================================
#Primeira pergunta: gráfico e horas dormidas, horas estudadas e nível de estresse
#===================================================================================
df["School_performance"] = (df["GPA"] -  df["GPA"].min()) / (df["GPA"].max() - df["GPA"].min())

conditions = [
    df["School_performance"] > 0.9,
    df["School_performance"] > 0.8,
    df["School_performance"] > 0.7,
    df["School_performance"] > 0.6,
    df["School_performance"] > 0.5,
    df["School_performance"] > 0.4,
    df["School_performance"] > 0.3,
    df["School_performance"] > 0.2,
    df["School_performance"] > 0.1
]
choices = ["A+", "A", "B+", "B", "C+", "C", "D+", "D", "E+"]

df["Performance_Category"] = np.select(conditions, choices, default="E")

stress_map = {"Low": 1, "Moderate": 2, "High": 3}
df["Stress_Level_Num"] = df["Stress_Level"].map(stress_map)

grouped = df.groupby("Performance_Category")[[
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "Stress_Level_Num"]].mean().reset_index()

# Gráfico de dispersão (cada ponto = uma categoria)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    grouped["Study_Hours_Per_Day"],
    grouped["Sleep_Hours_Per_Day"],
    s=(grouped["Stress_Level_Num"] ** 3) * 40,
    c=grouped["Stress_Level_Num"],
    cmap="RdYlGn_r",
    alpha=0.8,
    edgecolor="black"
)

# Adicionar rótulos das categorias
for i, row in grouped.iterrows():
    plt.text(
        row["Study_Hours_Per_Day"] + 0.025,
        row["Sleep_Hours_Per_Day"] + 0.025,
        row["Performance_Category"],
        fontsize=10,
        fontweight="bold"
    )

plt.title("Média de Horas de Estudo e Sono por Categoria de Performance", fontsize=14)
plt.xlabel("Média de Horas de Estudo por Dia", fontsize=12)
plt.ylabel("Média de Horas de Sono por Dia", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
cbar = plt.colorbar(scatter)
cbar.set_label("Nível Médio de Estresse", fontsize=12)
plt.tight_layout()
plt.show()

#===================================================================================
