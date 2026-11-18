# fuentes/datos_prep.py
import os
import re
import unicodedata
import pandas as pd

def generar_datos_iniciales(ruta_ini="datos/datos_ini/consultas_tres_lunas.csv"):
    """Genera un dataset realista de consultas de clientes de Tres Lunas Serenas."""
    os.makedirs(os.path.dirname(ruta_ini), exist_ok=True)
    
    datos_muestra = [
        # Categoría 0: Pedidos y Envíos
        ("¿Dónde está mi paquete? Lo pedí hace 5 días y no llega.", "Pedidos y Envíos"),
        ("Necesito cambiar la dirección de entrega de mi orden #4589.", "Pedidos y Envíos"),
        ("¿Cuánto tarda el envío exprés a Ciudad de México?", "Pedidos y Envíos"),
        ("Mi número de guía aparece como entregado pero no recibí nada.", "Pedidos y Envíos"),
        ("¿Hacen envíos internacionales o solo dentro de la república?", "Pedidos y Envíos"),
        ("El repartidor no encontró mi domicilio hoy, ¿cuándo vuelve?", "Pedidos y Envíos"),
        ("Quiero rastrear mi compra realizada ayer por la noche.", "Pedidos y Envíos"),
        ("¿Cuál es el costo de envío para compras mayores a $999?", "Pedidos y Envíos"),
        
        # Categoría 1: Devoluciones y Reembolsos
        ("El producto llegó roto, quiero solicitar mi reembolso.", "Devoluciones y Reembolsos"),
        ("¿Cuántos días tengo para devolver un artículo que no me gustó?", "Devoluciones y Reembolsos"),
        ("Ya envié el paquete de regreso, ¿cuándo se refleja el dinero?", "Devoluciones y Reembolsos"),
        ("Me enviaron un color equivocado, necesito hacer un cambio.", "Devoluciones y Reembolsos"),
        ("¿Cómo genero la etiqueta de devolución gratuita?", "Devoluciones y Reembolsos"),
        ("Cancelé mi pedido hace 3 días y aún veo el cargo en mi tarjeta.", "Devoluciones y Reembolsos"),
        ("La talla no me quedó bien, ¿puedo cambiarla por otra?", "Devoluciones y Reembolsos"),
        ("Quiero cancelar mi compra antes de que salga del almacén.", "Devoluciones y Reembolsos"),

        # Categoría 2: Información de Productos
        ("¿De qué material están hechas las velas aromáticas de lavanda?", "Información de Productos"),
        ("¿Cuándo volverán a tener stock del difusor lunar?", "Información de Productos"),
        ("¿Sus productos son libres de crueldad animal y veganos?", "Información de Productos"),
        ("¿Qué medidas tiene el tapete de meditación nocturna?", "Información de Productos"),
        ("¿Tienen disponible este set de té relajante para regalo?", "Información de Productos"),
        ("¿Cuál es la duración aproximada del incienso de sándalo?", "Información de Productos"),
        ("Busco recomendaciones para mejorar el descanso nocturno.", "Información de Productos"),
        ("¿Los aceites esenciales son 100% puros o vienen diluidos?", "Información de Productos"),

        # Categoría 3: Facturación y Cuenta
        ("Necesito factura fiscal de mi compra, ¿dónde ingreso mi RFC?", "Facturación y Cuenta"),
        ("Olvidé mi contraseña y no me llega el correo de recuperación.", "Facturación y Cuenta"),
        ("¿Qué métodos de pago aceptan en la tienda en línea?", "Facturación y Cuenta"),
        ("Mi cupón de bienvenida del 10% marca error al aplicarlo.", "Facturación y Cuenta"),
        ("Quiero actualizar mi correo electrónico registrado en mi perfil.", "Facturación y Cuenta"),
        ("El pago con tarjeta fue rechazado pero sí tengo saldo.", "Facturación y Cuenta"),
        ("¿Puedo pagar a meses sin intereses con tarjetas participantes?", "Facturación y Cuenta"),
        ("No encuentro mi historial de pedidos dentro de mi cuenta.", "Facturación y Cuenta"),
    ] * 8  # Multiplicamos para tener 256 registros balanceados para validación cruzada

    df = pd.DataFrame(datos_muestra, columns=["consulta_texto", "categoria"])
    # Introducir algunos nulos y duplicados intencionales para demostrar la limpieza
    df.loc[5, "consulta_texto"] = None
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    
    df.to_csv(ruta_ini, index=False, encoding="utf-8")
    print(f"[INFO] Dataset original guardado en: {ruta_ini} ({len(df)} registros)")

def limpiar_texto(texto):
    """Normaliza texto: minúsculas, sin acentos, sin caracteres especiales ni espacios extra."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    # Eliminar acentos
    texto = "".join(
        c for c in unicodedata.normalize("NFD", texto)
        if unicodedata.category(c) != "Mn"
    )
    # Conservar solo letras y espacios
    texto = re.sub(r"[^a-z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def limpiar_y_preparar(ruta_ini="datos/datos_ini/consultas_tres_lunas.csv",
                       ruta_limp="datos/datos_limp/consultas_limpias.csv"):
    """Aplica reglas de negocio, limpieza de nulos/duplicados y codificación de categorías."""
    if not os.path.exists(ruta_ini):
        generar_datos_iniciales(ruta_ini)

    df = pd.read_csv(ruta_ini)

    # 1. Eliminar registros con texto nulo o vacío
    df = df.dropna(subset=["consulta_texto", "categoria"]).copy()

    # 2. Normalizar texto
    df["texto_limpio"] = df["consulta_texto"].apply(limpiar_texto)
    df = df[df["texto_limpio"] != ""]

    # 3. Eliminar duplicados exactos tras normalización
    df = df.drop_duplicates(subset=["texto_limpio"]).reset_index(drop=True)

    # 4. Codificar variable objetivo (Categoría -> ID numérico)
    categorias_ordenadas = sorted(df["categoria"].unique())
    mapa_categorias = {cat: idx for idx, cat in enumerate(categorias_ordenadas)}
    df["categoria_id"] = df["categoria"].map(mapa_categorias)

    os.makedirs(os.path.dirname(ruta_limp), exist_ok=True)
    df.to_csv(ruta_limp, index=False, encoding="utf-8")
    print(f"[INFO] Dataset limpio guardado en: {ruta_limp} ({len(df)} registros únicos)")
    return df, mapa_categorias

if __name__ == "__main__":
    limpiar_y_preparar()