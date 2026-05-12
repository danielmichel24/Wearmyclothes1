import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from modelo import generar_embedding
from utils import cosine_similarity


def registrar_prendas(prendas, mostrar_logs=True):
    """
    Recibe una lista de prendas del usuario y genera su embedding.

    Entrada esperada:
    [
        {
            "id": 1,
            "url": "https://drive.google.com/uc?id=..."
        },
        {
            "id": 2,
            "url": "https://drive.google.com/uc?id=..."
        }
    ]

    Retorna:
    [
        {
            "id": 1,
            "url": "...",
            "embedding": [...]
        }
    ]
    """

    prendas_con_embeddings = []

    for prenda in prendas:
        prenda_id = prenda.get("id")
        url = prenda.get("url")

        if prenda_id is None or url is None:
            if mostrar_logs:
                print("Prenda inválida. Debe tener 'id' y 'url'.")
            continue

        if mostrar_logs:
            print(f"Registrando prenda ID {prenda_id}")

        embedding = generar_embedding(url)

        if embedding is None:
            if mostrar_logs:
                print(f"No se pudo registrar la prenda ID {prenda_id}")
            continue

        prendas_con_embeddings.append({
            "id": prenda_id,
            "url": url,
            "embedding": embedding
        })

        if mostrar_logs:
            print(f"Prenda ID {prenda_id} registrada correctamente")

    return prendas_con_embeddings


def escanear_prenda(imagen_a_escanear, conjunto_de_prendas, umbral=0.70, mostrar_logs=True):
    """
    Escanea una imagen y la compara contra las prendas registradas.

    imagen_a_escanear puede ser:
    - URL
    - ruta local
    - bytes de imagen

    conjunto_de_prendas debe ser la salida de registrar_prendas().

    Retorna:
    {
        "detectado": True,
        "prenda_id": 1,
        "similitud": 0.7368,
        "top_matches": [
            {"id": 1, "score": 0.7368},
            {"id": 2, "score": 0.4775}
        ],
        "mensaje": "Prenda detectada correctamente."
    }
    """

    embedding_nuevo = generar_embedding(imagen_a_escanear)

    if embedding_nuevo is None:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": 0,
            "top_matches": [],
            "mensaje": "No se pudo generar el embedding de la imagen enviada."
        }

    mejor_similitud = -1
    mejor_prenda_id = None
    resultados = []

    for prenda in conjunto_de_prendas:
        sim = cosine_similarity(embedding_nuevo, prenda["embedding"])

        resultados.append({
            "id": prenda["id"],
            "score": round(float(sim), 4)
        })

        if mostrar_logs:
            print(f"Comparando con prenda ID {prenda['id']}: {sim:.4f}")

        if sim > mejor_similitud:
            mejor_similitud = sim
            mejor_prenda_id = prenda["id"]

    resultados = sorted(
        resultados,
        key=lambda x: x["score"],
        reverse=True
    )

    if mostrar_logs:
        print("Mejor similitud:", mejor_similitud)
        print("Mejor prenda ID:", mejor_prenda_id)

    if mejor_similitud < umbral:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": round(float(mejor_similitud), 4),
            "top_matches": resultados[:5],
            "mensaje": "No se encontró una prenda suficientemente parecida."
        }

    return {
        "detectado": True,
        "prenda_id": mejor_prenda_id,
        "similitud": round(float(mejor_similitud), 4),
        "top_matches": resultados[:5],
        "mensaje": "Prenda detectada correctamente."
    }


def ejecutar_prueba():
    """
    Función solo para pruebas locales.

    Esta función NO se ejecuta cuando Eduardo importe este archivo.
    Solo se ejecuta cuando corres directamente:

    python main.py
    """

    prendas_usuario = [
        {
            #gorra bulls 1
            "id": 1,
            "url": "https://drive.google.com/uc?id=1U1MIGL-voE901McRb-ydu9Xm3Sj7flMH"
        },
        {
            #gorra bulls 2
           "id": 2,
            "url": "https://drive.google.com/uc?id=1N0wZwx5Xc1_bj_XrNg0ShGIJ-FU-WDiZ"
        },
        {
            #pantalon azul
            "id": 3,
            "url": "https://drive.google.com/uc?id=1eIckdPm4bDodOw4D1VNvQ31vCggl0Y5x"
        },
        {
            #Sueter verde 1
            "id": 4,
            "url": "https://drive.google.com/uc?id=1Hb8dcYTrzQK1OQu8c94B4ZJqGTP4jt_G"
        },
        {
            #camisa vaquera de cuadros
            "id": 5,
            "url": "https://drive.google.com/uc?id=1qetaadJSRr48qaOhkQ8hkfMrRMP0DTMB"
        },    
        {
            #camisa lisa gris 3xl
            "id": 6,
            "url": "https://drive.google.com/uc?id=1WFli8HFXq-txJ-WYtr8sxUHGrNEy4Zat"
        },
        {
            #camisa de cuadros mixta
            "id": 7,
            "url": "https://drive.google.com/uc?id=1KpCoanICuOuX7y4HOPI6QYr_31s1Mhhj"
        },
        {
            #camisa negra de botones
            "id": 7,
            "url": "https://drive.google.com/uc?id=1JoncWFE0r-SWbKXs4H8oVYw2_MWChSQa"
        },
        {
            #camisa negra lisa
            "id": 8,
            "url": "https://drive.google.com/uc?id=1NdX3LXPgqBE9PryVl4NjxBMcDI588Y7G"
        },
        {
            #sudadera negra sin flash
            "id": 9,
            "url": "https://drive.google.com/uc?id=1le9IUh_qgu2K56w08Rcmi0y4BhZoIEvu"
        },

        
    ]

    #su
    imagen_a_escanear ="https://drive.google.com/uc?id=1SCjgX_R80ks6oZIzqY1wbk98kDeEIcED"

    prendas_con_embeddings = registrar_prendas(
        prendas=prendas_usuario,
        mostrar_logs=True
    )

    resultado = escanear_prenda(
        imagen_a_escanear=imagen_a_escanear,
        conjunto_de_prendas=prendas_con_embeddings,
        umbral=0.70,
        mostrar_logs=True
    )

    print("\nResultado final:")
    print(json.dumps(resultado, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    ejecutar_prueba()