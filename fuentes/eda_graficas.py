# fuentes/eda_graficas.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_limp = pd.read_csv("datos/datos_limp/student_uci_limpio.csv")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Gráfica 1: Distribución de estudiantes por nivel de intervención
orden = ["Nivelación Intensiva", "Tutoría Focalizada", "Enriquecimiento Académico"]
sns.countplot(
    data=df_limp, y="nivel_intervencion", hue="nivel_intervencion",
    order=orden, palette="Blues_r", legend=False, ax=axes[0]
)
axes[0].set_title("Distribución de Estudiantes por Nivel de Intervención\n(Student Performance Dataset - UCI)", fontweight="bold")
axes[0].set_xlabel("Número de estudiantes")
axes[0].set_ylabel("")
for container in axes[0].containers:
    axes[0].bar_label(container, padding=4)

# Gráfica 2: Distribución de Calificación Final (G3) por Nivel de Intervención
sns.boxplot(
    data=df_limp, x="nivel_intervencion", y="G3",
    hue="nivel_intervencion", order=orden, palette="Blues_r", legend=False, ax=axes[1]
)
axes[1].set_title("Distribución de Calificación Final (G3) por Grupo", fontweight="bold")
axes[1].set_xlabel("")
axes[1].set_ylabel("Calificación Final G3 (0 - 20)")
axes[1].tick_params(axis="x", rotation=15)

os.makedirs("evidencias", exist_ok=True)
plt.tight_layout()
plt.savefig("evidencias/eda_student_uci.png", dpi=200)
print("[INFO] Gráfica guardada en: evidencias/eda_student_uci.png")