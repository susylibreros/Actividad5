import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar dataset original y dataset limpio
df_ini = pd.read_csv("datos/datos_ini/consultas_tres_lunas.csv").dropna(subset=["consulta_texto"])
df_limp = pd.read_csv("datos/datos_limp/consultas_limpias.csv")

# 2. Calcular longitud de caracteres antes y después de limpiar
df_ini["longitud_caracteres"] = df_ini["consulta_texto"].astype(str).apply(len)
df_limp["longitud_caracteres"] = df_limp["texto_limpio"].astype(str).apply(len)

# 3. Configurar estilo y lienzo con 2 subgráficas lado a lado
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Gráfica A: Distribución de las 4 categorías (Datos Limpios)
orden_cats = df_limp["categoria"].value_counts().index
sns.countplot(
    data=df_limp,
    y="categoria",
    hue="categoria",
    order=orden_cats,
    palette="Purples_r",
    legend=False,
    ax=axes[0]
)
axes[0].set_title("Distribución de Consultas por Categoría\n(Tres Lunas Serenas - Datos Limpios)", fontweight="bold")
axes[0].set_xlabel("Cantidad de consultas únicas")
axes[0].set_ylabel("")

# Agregar etiquetas numéricas al final de cada barra
for container in axes[0].containers:
    axes[0].bar_label(container, padding=4)

# Gráfica B: Longitud de caracteres Antes vs. Después de limpiar
sns.histplot(
    df_ini["longitud_caracteres"],
    color="#B8B8D1",
    label="Original (con signos/acentos)",
    kde=True,
    bins=10,
    ax=axes[1]
)
sns.histplot(
    df_limp["longitud_caracteres"],
    color="#5B2C6F",
    label="Limpio (normalizado)",
    kde=True,
    bins=10,
    ax=axes[1]
)
axes[1].set_title("Longitud de Consultas (Caracteres)\nAntes vs. Después de la Limpieza", fontweight="bold")
axes[1].set_xlabel("Número de caracteres")
axes[1].set_ylabel("Frecuencia")
axes[1].legend()

plt.tight_layout()
plt.savefig("eda_tres_lunas.png", dpi=200)
print("[INFO] Gráfica guardada exitosamente como: eda_tres_lunas.png")
plt.show()
