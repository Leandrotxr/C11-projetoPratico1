import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../data/student_lifestyle_dataset.csv', delimiter=',')

print("Dataset carregado com sucesso!\n")

df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
df = df.replace(r'^\s*$', np.nan, regex=True)

#===================================================================================
#Primeira pergunta: horas dormidas e nível de estresse
#===================================================================================
mean_sleep_by_stress = df.groupby('Stress_Level')['Sleep_Hours_Per_Day'].mean().reindex(['Low', 'Moderate', 'High'])
mean_sleep_by_stress = mean_sleep_by_stress.dropna()
# Plotar gráfico de barras
plt.figure(figsize=(8,5))
mean_sleep_by_stress.plot(kind='bar', color=['green', 'orange', 'red'])
plt.title('Média de Horas de Sono por Nível de Estresse')
plt.xlabel('Nível de Estresse')
plt.ylabel('Horas de Sono por Dia')
plt.xticks(rotation=0)
plt.ylim(0, max(mean_sleep_by_stress) + 1)
plt.grid(axis='y', linestyle='', alpha=0.7)
plt.show()

#===================================================================================
#Segunda pergunta: horas dormidas, atividades fisicas e estresse
#===================================================================================
df["School_performance"] = (df["GPA"] -  df["GPA"].min()) / (df["GPA"].max() - df["GPA"].min())
stress_map = {"Low": 1, "Moderate": 2, "High": 3}
df["Stress_Level_Num"] = df["Stress_Level"].map(stress_map)

sleep_mask = df["Sleep_Hours_Per_Day"].between(7, 8)
activity_mask = df["Physical_Activity_Hours_Per_Day"] > df["Physical_Activity_Hours_Per_Day"].median()
health_group = df[sleep_mask & activity_mask].sample(n=100, random_state=42)

low_sleep_mask = df["Sleep_Hours_Per_Day"] <= df["Sleep_Hours_Per_Day"].nsmallest(500).max()
activity_mask = df["Physical_Activity_Hours_Per_Day"] < df["Physical_Activity_Hours_Per_Day"].median()
not_health_group = df[low_sleep_mask & activity_mask].sample(n=100, random_state=42)

media_saudavel = health_group["Stress_Level_Num"].mean()
media_nao_saudavel = not_health_group["Stress_Level_Num"].mean()

grupos = ["Saudável", "Não Saudável"]
medias = [media_saudavel, media_nao_saudavel]
plt.figure(figsize=(6, 5))
barras = plt.bar(grupos, medias, color=["green", "red"], alpha=0.9)
plt.title("Comparação do Nível Médio de Estresse entre Grupos", fontsize=10)
plt.ylabel("Nível Médio de Estresse")
plt.ylim(0, 3)

for barra in barras:
    plt.text(
        barra.get_x() + barra.get_width() / 2,
        barra.get_height() / 2,
        f"{barra.get_height():.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )

plt.grid(axis="y", linestyle="", alpha=0.6)
plt.tight_layout()
plt.show()

#===================================================================================
#Terceira pergunta: gráfico e horas dormidas, horas estudadas e nível de estresse
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
#Quarta pergunta: Distribuição diária média dos estudantes
#===================================================================================
df["Free_Hours_Per_Day"] = 24 - (df["Study_Hours_Per_Day"] + df["Extracurricular_Hours_Per_Day"] + df["Sleep_Hours_Per_Day"] + df["Physical_Activity_Hours_Per_Day"])

mean_day = df[[
    "Study_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Free_Hours_Per_Day"
]].mean()

labels = {
    "Study_Hours_Per_Day": "Horas de Estudo",
    "Sleep_Hours_Per_Day": "Horas de Sono",
    "Physical_Activity_Hours_Per_Day": "Atividade Física",
    "Extracurricular_Hours_Per_Day": "Atividades Extracurriculares",
    "Free_Hours_Per_Day": "Horas Livres"
}

mean_day.index = mean_day.index.map(labels)

plt.figure(figsize=(7,7))

def label_hours(pct, values):
    percent = pct/100 * sum(values)
    return f"{percent:.1f}h"

plt.pie(
    mean_day,
    labels=mean_day.index,
    autopct=lambda percent: label_hours(percent, mean_day),
)

plt.title("Distribuição Média das 24h do Dia (em horas)", fontsize=14)
plt.tight_layout()
plt.show()

