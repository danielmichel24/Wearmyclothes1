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

def escanearPrenda(imagen_nueva, base_datos):
    embedding_nuevo = generar_embedding(imagen_nueva)

    mejor_similitud = -1
    mejor_prenda = None

    for prenda in base_datos:
        sim = cosine_similarity(embedding_nuevo, prenda["embedding"])

        print(f"Comparando con {prenda['nombre']}: {sim:.4f}")

        if sim > mejor_similitud:
            mejor_similitud = sim
            mejor_prenda = prenda["nombre"]

    print("Mejor similitud:", mejor_similitud)

    if mejor_similitud < 0.90:
        return "no detectado"

    return mejor_prenda

from modelo import generar_embedding

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





# 👇 TUS IMÁGENES
imagenes = [
    {"nombre": "Camisa con botones", "ruta": "imagenes_prueba/camisa1_1.webp"},
    {"nombre": "Camisa lisa", "ruta": "imagenes_prueba/camisaazul.webp"}
]

# 🔥 SOLO UNA VEZ
base_datos = registrar_prendas(imagenes)


resultado = escanearPrenda("imagenes_prueba/test.webp", base_datos) 

print("\nResultado final:", resultado)