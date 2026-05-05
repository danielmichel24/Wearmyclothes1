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


# TUS IMÁGENES
imagenes = [
    {"nombre": "Camisa con botones", "ruta": "imagenes_prueba/camisa1_1.webp"},
    {"nombre": "Camisa lisa", "ruta": "imagenes_prueba/camisaazul.webp"}
]

#  SOLO UNA VEZ
base_datos = registrar_prendas(imagenes)

#vahora con umbral dinámico
resultado = escanear_prenda(
    "imagenes_prueba/test.webp",
    base_datos,
    0.8
)

print("\nResultado final:", resultado)