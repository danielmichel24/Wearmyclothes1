import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from PIL import Image
import numpy as np
from modelo import generar_embedding
from utils import cosine_similarity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils import cosine_similarity



# FUNCIÓN PRINCIPAL
def escanear_prenda(imagen, conjunto_de_imagenes, umbral):
    embedding_nuevo = generar_embedding(imagen)

    if embedding_nuevo is None:
        return {
            "detectado": False,
            "prenda": None,
            "similitud": 0
        }

    mejor_similitud = -1
    mejor_prenda = None

    for prenda in conjunto_de_imagenes:
        sim = cosine_similarity(embedding_nuevo, prenda["embedding"])

        print(f"Comparando con {prenda['nombre']}: {sim:.4f}")

        if sim > mejor_similitud:
            mejor_similitud = sim
            mejor_prenda = prenda["nombre"]

    print("Mejor similitud:", mejor_similitud)

    if mejor_similitud < umbral:
        return {
            "detectado": False,
            "prenda": None,
            "similitud": float(mejor_similitud)
        }

    return {
        "detectado": True,
        "prenda": mejor_prenda,
        "similitud": float(mejor_similitud)
    }


# REGISTRAR PRENDAS
def registrar_prendas(imagenes):
    base_datos = []

    for prenda in imagenes:
        embedding = generar_embedding(prenda["ruta"])

        if embedding is None:
            continue

        base_datos.append({
            "nombre": prenda["nombre"],
            "embedding": embedding
        })

    return base_datos


# CARGAR IMÁGENES AUTOMÁTICAMENTE
def cargar_imagenes_desde_carpeta(ruta_carpeta):
    imagenes = []

    for archivo in os.listdir(ruta_carpeta):
        if archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):

            # ignorar test
            if archivo.lower().startswith("test"):
                continue

            nombre_base = archivo.split('.')[0]
            nombre_base = ''.join([c for c in nombre_base if not c.isdigit()]).replace('_', ' ').strip()

            imagenes.append({
                "nombre": nombre_base,
                "ruta": os.path.join(ruta_carpeta, archivo)
            })

    return imagenes


#BUSCAR IMAGEN DE TEST AUTOMÁTICAMENTE
def obtener_ruta_test(ruta_carpeta):
    for archivo in os.listdir(ruta_carpeta):
        if archivo.lower().startswith("test") and archivo.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            return os.path.join(ruta_carpeta, archivo)

    return None



# EJECUCIÓN

imagenes = cargar_imagenes_desde_carpeta("imagenes_prueba")

base_datos = registrar_prendas(imagenes)

ruta_test = obtener_ruta_test("imagenes_prueba")

if ruta_test is None:
    print("❌ No se encontró imagen de test")
else:
    resultado = escanear_prenda(
        ruta_test,
        base_datos,
        0.8
    )

    print("\nResultado final:", resultado)