import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

df = pd.read_csv('../data/student_lifestyle_dataset.csv', delimiter=',')

print("Dataset carregado com sucesso!\n")
print(df.head())
print(df.info())

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

#===================================================================================
#Quinta pergunta: Relação entre GPA e quantidade de horas estudadas por dia
#===================================================================================
df["GPA_rounded"] = (df["GPA"] / 0.25).round() * 0.25
avg_per_group = df.groupby("GPA_rounded")["Study_Hours_Per_Day"].mean().reset_index()

plt.scatter(avg_per_group["GPA_rounded"], avg_per_group["Study_Hours_Per_Day"], s=100, label="Average per group")
plt.title("Relação entre GPA e horas de estudo por dia")
plt.xlabel("GPA")
plt.ylabel("Horas de Estudo por Dia")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

#===================================================================================
#Sexta pergunta: Comparação entre horas pessoais e não pessoais por nível de estresse
#===================================================================================
df["Personal_Hours_Per_Day"] = df["Social_Hours_Per_Day"] + df["Physical_Activity_Hours_Per_Day"]
df["Non_Personal_Hours_Per_Day"] = df["Study_Hours_Per_Day"] + df["Extracurricular_Hours_Per_Day"]
personal_hours_stress_group = df.groupby("Stress_Level")["Personal_Hours_Per_Day"].mean()
non_personal_hours_stress_group = df.groupby("Stress_Level")["Non_Personal_Hours_Per_Day"].mean()

order = ["Low", "Moderate", "High"]
personal_hours_stress_group = personal_hours_stress_group.reindex(order)
non_personal_hours_stress_group = non_personal_hours_stress_group.reindex(order)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(personal_hours_stress_group.index, personal_hours_stress_group.values)
plt.title("Relação de quantidade de horas pessoais por nível de estresse")
plt.xlabel("Nível de estresse")
plt.ylabel("Horas sociais e de atividade física por dia")

plt.subplot(1, 2, 2)
plt.bar(non_personal_hours_stress_group.index, non_personal_hours_stress_group.values)
plt.title("Relação de quantidade de horas não pessoais por nível de estresse")
plt.xlabel("Nível de estresse")
plt.ylabel("Horas de estudo e atividades extracurriculares por dia")

plt.tight_layout()
plt.show()

#===================================================================================
#Sétima pergunta: relação entre horas de estudo e performace acadêmica (interpolada)
#===================================================================================
from scipy.interpolate import make_interp_spline

study_perf = df.groupby("Study_Hours_Per_Day")["GPA"].mean().reset_index().sort_values("Study_Hours_Per_Day")

x = study_perf["Study_Hours_Per_Day"]
y = study_perf["GPA"]
x_smooth = np.linspace(x.min(), x.max(), 1000)
spline = make_interp_spline(x, y)
y_smooth = spline(x_smooth)

plt.figure(figsize=(10,6))
plt.plot(x_smooth, y_smooth, color="blue", linewidth=2, label="Interpolação (tendência)")
plt.scatter(x, y, color="red", alpha=0.7, label="Média real por grupo")

plt.title("Relação entre Horas de Estudo e Performance Acadêmica (Interpolada)", fontsize=14)
plt.xlabel("Horas de Estudo por Dia")
plt.ylabel("GPA Médio")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

#===================================================================================
#Oitava pergunta: relação entre horas livres e performance acadêmica
#===================================================================================
free_perf = df.groupby("Performance_Category")["Free_Hours_Per_Day"].mean().reindex(choices + ["E"])

plt.figure(figsize=(8,5))
plt.bar(free_perf.index, free_perf.values, color="green", alpha=0.8)
plt.title("Média de Horas Livres por Categoria de Performance", fontsize=14)
plt.xlabel("Categoria de Performance Acadêmica")
plt.ylabel("Horas Livres por Dia")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

#===================================================================================
#Nona pergunta: Distribuição Gaussiana do GPA em relação às horas de sono
#===================================================================================
bins = [0, 4, 6, 8, 10, 24]
labels = ["0–4h", "4–6h", "6–8h", "8–10h", "10h+"]
df["Sleep_rates"] = pd.cut(df["Sleep_Hours_Per_Day"], bins=bins, labels=labels, include_lowest=True)

plt.figure(figsize=(10, 6))

colors = ["#ff9999", "#ffcc66", "#99ff99", "#66b3ff", "#cc99ff"]

for faixa, cor in zip(labels, colors):
    subset = df[df["Sleep_rates"] == faixa]["GPA"].dropna()
    if subset.empty:
        continue

    mu, sigma = subset.mean(), subset.std()

    x = np.linspace(subset.min(), subset.max(), 200)
    y = norm.pdf(x, mu, sigma)

    y_scaled = y * (len(subset) * (subset.max() - subset.min()) / 20)

    plt.hist(subset, bins=20, alpha=0.35, color=cor, label=faixa)
    plt.plot(x, y_scaled, color=cor, linewidth=2)

plt.title("Distribuição Gaussiana do GPA por Faixa de Horas de Sono", fontsize=14)
plt.xlabel("GPA", fontsize=12)
plt.ylabel("Número de Alunos por Faixa de GPA", fontsize=12)
plt.legend(title="Faixas de Sono", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

#===================================================================================
#Décima pergunta: Relação entre sono, atividade física e desempenho acadêmico
#===================================================================================
sleep_hours = df["Sleep_Hours_Per_Day"]
physical_hours = df["Physical_Activity_Hours_Per_Day"]

sleep_small_amount_hours = sleep_hours < 6
sleep_ideal_amount_hours = (sleep_hours >= 6) & (sleep_hours <= 8)
sleep_alot_amount_hours = sleep_hours > 8
sleep_category = pd.Series(index=df.index, dtype="object")
sleep_category[sleep_small_amount_hours] = "Pouco Sono"
sleep_category[sleep_ideal_amount_hours] = "Sono Ideal"
sleep_category[sleep_alot_amount_hours] = "Muito Sono"
df["Sleep_Category"] = sleep_category

avg_activity = df["Physical_Activity_Hours_Per_Day"].median()
physical_small_amount_hours = physical_hours <= avg_activity
physical_alot_amount_hours = physical_hours > avg_activity
avg_activity_category = pd.Series(index=df.index, dtype="object")
avg_activity_category[physical_small_amount_hours] = "Pouca atividade"
avg_activity_category[physical_alot_amount_hours] = "Muita atividade"
df["Physical_Category"] = avg_activity_category

grouped = df.groupby(["Sleep_Category", "Physical_Category"])["GPA"].mean()
heatmap_data = grouped.unstack()
sleep_order = ["Pouco Sono", "Sono Ideal", "Muito Sono"]
heatmap_data = heatmap_data.reindex(sleep_order)

plt.figure(figsize=(8, 5))
plt.imshow(heatmap_data, cmap="YlGnBu", aspect="auto")
plt.colorbar(label="GPA Médio")
plt.xticks(ticks=np.arange(len(heatmap_data.columns)), labels=heatmap_data.columns)
plt.yticks(ticks=np.arange(len(heatmap_data.index)), labels=heatmap_data.index)
plt.title("GPA Médio por Categoria de Sono e Atividade Física", fontsize=14)
plt.xlabel("Nível de Atividade Física")
plt.ylabel("Categoria de Sono")
plt.tight_layout()
plt.show()
