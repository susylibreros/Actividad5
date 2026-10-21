import pandas as pd
import numpy as np
import re

def limpiar_texto(texto: str) -> str:
    """Aplica minúsculas, remueve caracteres especiales y espacios extra."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def preparar_datos_consultas(input_path: str, output_path: str) -> pd.DataFrame:
    """Lee datos crudos, maneja nulos, limpia texto, mapea etiquetas y guarda el dataset limpio."""
    df = pd.read_csv(input_path)
    
    # Manejo de nulos
    df = df.dropna(subset=['text', 'label']).copy()
    
    # Aplicar limpieza
    df['text_clean'] = df['text'].apply(limpiar_texto)
    
    # Asegurar tipos de datos
    df['label'] = df['label'].astype(int)
    
    # Guardar versión limpia
    df.to_csv(output_path, index=False)
    print(f"✅ Datos procesados y guardados exitosamente en: {output_path}")
    return df

if __name__ == "__main__":
    preparar_datos_consultas(
        input_path="../datos/datos_ini/consultas_clientes_raw.csv",
        output_path="../datos/datos_limp/consultas_clientes_clean.csv"
    )
