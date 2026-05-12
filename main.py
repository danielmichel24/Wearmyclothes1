import os
import json
from modelo import generar_embedding
from utils import cosine_similarity
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# FUNCIÓN PRINCIPAL
def escanear_prenda(imagen_a_escanear, conjunto_de_prendas, umbral=0.70):
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

        print(f"Comparando con prenda ID {prenda['id']}: {sim:.4f}")

        if sim > mejor_similitud:
            mejor_similitud = sim
            mejor_prenda_id = prenda["id"]

    resultados = sorted(
        resultados,
        key=lambda x: x["score"],
        reverse=True
    )

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

# REGISTRAR PRENDAS
def registrar_prendas(prendas):
    """
    Recibe una lista de objetos con id y url.
    Genera el embedding de cada imagen y lo guarda asociado al id.

    Ejemplo de entrada:
    [
        {
            "id": 3,
            "url": "https://drive.usercontent.google.com/download?id=1eIckdPm4bDodOw4D1VNvQ31vCggl0Y5x&authuser=0"
        },
        {
            "id": 6,
            "url": "https://drive.usercontent.google.com/download?id=1eIckdPm4bDodOw4D1VNvQ31vCggl0Y5x&authuser=0"
        }
        {
            "id": 1
            "url" :"https://drive.google.com/file/d/1U1MIGL-voE901McRb-ydu9Xm3Sj7flMH/view?usp=drive_link"       
        }
        {
            "id": 2
            "url" :"https://drive.google.com/file/d/1N0wZwx5Xc1_bj_XrNg0ShGIJ-FU-WDiZ/view?usp=sharing"       
        }
    ]
    """

    base_datos = []

    for prenda in prendas:
        print(f"Registrando prenda ID {prenda['id']}")

        embedding = generar_embedding(prenda["url"])

        if embedding is None:
            print(f"No se pudo registrar la prenda ID {prenda['id']}")
            continue

        base_datos.append({
            "id": prenda["id"],
            "url": prenda["url"],
            "embedding": embedding
        })

        print(f"Prenda ID {prenda['id']} registrada correctamente")

    return base_datos


# EJECUCIÓN DE PRUEBA
if __name__ == "__main__":

    prendas_usuario = [
        {
            #pantalon Mezclilla azul
            "id": 3,
            "url": "https://drive.google.com/uc?id=1eIckdPm4bDodOw4D1VNvQ31vCggl0Y5x"
        },
        {
            #Camisa de Cuadros vaquera
            "id": 6,
            "url": "https://drive.google.com/uc?id=1qetaadJSRr48qaOhkQ8hkfMrRMP0DTMB"
        },
        {
            #Gorra Bulls 1
            "id": 1,
            "url": "https://drive.google.com/uc?id=1U1MIGL-voE901McRb-ydu9Xm3Sj7flMH"
        },
        {
            #Gorra One piece 2
            "id": 2,
            "url": "https://drive.google.com/uc?id=1xFLxg6PDGU6JpAzU6QgsnTW7bLegMYvc"
        },
        {
            #sueter verde 1
            "id": 4,
            "url": "https://drive.google.com/uc?export=download&id=1Hb8dcYTrzQK1OQu8c94B4ZJqGTP4jt_G"       
        }
    ]

    with open("imagenes_prueba/test.jpeg", "rb") as archivo:
        imagen_a_escanear = archivo.read()

    base_datos = registrar_prendas(prendas_usuario)

    resultado = escanear_prenda(
        imagen_a_escanear=imagen_a_escanear,
        conjunto_de_prendas=base_datos,
        umbral=0.70
    )

    print("\nResultado final:")
    print(json.dumps(resultado, indent=4, ensure_ascii=False))