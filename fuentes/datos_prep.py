# fuentes/datos_prep.py
import os
import io
import zipfile
import urllib.request
import pandas as pd
from sklearn.preprocessing import StandardScaler

def descargar_dataset_uci(ruta_ini="datos/datos_ini/student-mat.csv"):
    """Descarga el Student Performance Dataset desde UCI Machine Learning Repository."""
    if os.path.exists(ruta_ini):
        print(f"[INFO] Dataset público encontrado en: {ruta_ini}")
        return

    os.makedirs(os.path.dirname(ruta_ini), exist_ok=True)
    url_uci = "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip"
    print("[INFO] Descargando Student Performance Dataset desde UCI ML Repository...")

    req = urllib.request.urlopen(url_uci)
    with zipfile.ZipFile(io.BytesIO(req.read())) as z:
        with z.open("student-mat.csv") as f_in, open(ruta_ini, "wb") as f_out:
            f_out.write(f_in.read())

    print(f"[INFO] Dataset UCI guardado exitosamente en: {ruta_ini}")

def limpiar_y_preparar(ruta_ini="datos/datos_ini/student-mat.csv",
                       ruta_limp="datos/datos_limp/student_uci_limpio.csv"):
    """Limpia, codifica y estandariza el Student Performance Dataset (UCI)."""
    descargar_dataset_uci(ruta_ini)

    # El dataset oficial de UCI usa punto y coma (';') como separador
    df = pd.read_csv(ruta_ini, sep=";")

    # 1. Eliminar duplicados y valores nulos
    df = df.drop_duplicates().dropna().copy()

    # 2. Reglas de negocio: acotar notas [0, 20] y ausencias >= 0
    for col_nota in ["G1", "G2", "G3"]:
        df[col_nota] = df[col_nota].clip(0, 20)
    df["absences"] = df["absences"].clip(lower=0)

    # 3. Creación de la variable objetivo multiclase a partir de G3
    def clasificar_intervencion(g3):
        if g3 < 10:
            return "Nivelación Intensiva"
        elif g3 <= 13:
            return "Tutoría Focalizada"
        else:
            return "Enriquecimiento Académico"

    df["nivel_intervencion"] = df["G3"].apply(clasificar_intervencion)
    mapa_clases = {
        "Nivelación Intensiva": 0,
        "Tutoría Focalizada": 1,
        "Enriquecimiento Académico": 2
    }
    df["intervencion_id"] = df["nivel_intervencion"].map(mapa_clases)

    # 4. Codificación de variables categóricas binarias (yes/no -> 1/0)
    cols_binarias = ["schoolsup", "famsup", "paid", "higher", "internet"]
    for col in cols_binarias:
        df[f"{col}_bin"] = (df[col].str.lower() == "yes").astype(int)

    # 5. Estandarización Z-score de variables numéricas predictoras (sin incluir G3 para evitar data leakage)
    cols_numericas = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "absences", "G1", "G2"]
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[cols_numericas]),
        columns=[f"{c}_std" for c in cols_numericas]
    )

    cols_finales_bin = [f"{c}_bin" for c in cols_binarias]
    df_limpio = pd.concat([
        df[cols_finales_bin + ["G3", "nivel_intervencion", "intervencion_id"]].reset_index(drop=True),
        df_scaled
    ], axis=1)

    os.makedirs(os.path.dirname(ruta_limp), exist_ok=True)
    df_limpio.to_csv(ruta_limp, index=False, encoding="utf-8")
    print(f"[INFO] Dataset limpio guardado en: {ruta_limp} ({len(df_limpio)} registros)")
    return df_limpio, mapa_clases

if __name__ == "__main__":
    limpiar_y_preparar()