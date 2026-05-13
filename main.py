import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from modelo import generar_embedding, normalizar_url_google_drive
from utils import cosine_similarity, formatear_porcentaje, normalizar_umbral


def _obtener_url_referencia(referencia):
    """
    Obtiene la URL desde distintos nombres posibles.

    Esto permite recibir referencias como:
    - "https://drive.google.com/uc?id=..."
    - {"id": 1, "url": "https://drive.google.com/uc?id=..."}
    - {"id": 1, "image_url": "..."}
    - {"id": 1, "imagen_url": "..."}
    """

    if isinstance(referencia, str):
        return referencia

    if isinstance(referencia, dict):
        return (
            referencia.get("url")
            or referencia.get("image_url")
            or referencia.get("imagen_url")
            or referencia.get("drive_url")
        )

    return None


def _normalizar_referencia(referencia, indice):
    """
    Normaliza una referencia para que internamente tenga una estructura uniforme.
    """

    if isinstance(referencia, str):
        url = normalizar_url_google_drive(referencia)
        return {
            "id": indice + 1,
            "url": url,
            "embedding": None,
            "metadata": {}
        }

    if isinstance(referencia, dict):
        url_original = _obtener_url_referencia(referencia)
        url = normalizar_url_google_drive(url_original) if url_original else None

        return {
            "id": referencia.get("id", indice + 1),
            "url": url,
            "embedding": referencia.get("embedding"),
            "metadata": {
                key: value
                for key, value in referencia.items()
                if key not in {"id", "url", "image_url", "imagen_url", "drive_url", "embedding"}
            }
        }

    return {
        "id": indice + 1,
        "url": None,
        "embedding": None,
        "metadata": {}
    }


def registrar_prendas(prendas, mostrar_logs=True):
    """
    Recibe una lista de prendas del usuario y genera sus embeddings.

    Entrada esperada:
    [
        {
            "id": 1,
            "url": "https://drive.google.com/uc?id=..."
        }
    ]

    Tambien acepta URLs normales de Google Drive tipo:
    https://drive.google.com/file/d/ID/view?usp=drivesdk

    Retorna:
    [
        {
            "id": 1,
            "url": "https://drive.google.com/uc?id=...",
            "embedding": <numpy.ndarray>
        }
    ]

    Esta funcion sirve si quieres precalcular las referencias una vez y luego
    llamar escanear_prenda() varias veces sin regenerar embeddings.
    """

    prendas_con_embeddings = []

    for indice, prenda in enumerate(prendas):
        referencia = _normalizar_referencia(prenda, indice)
        prenda_id = referencia["id"]
        url = referencia["url"]

        if prenda_id is None or url is None:
            if mostrar_logs:
                print("Prenda invalida. Debe tener 'id' y 'url'.")
            continue

        if mostrar_logs:
            print(f"Registrando prenda ID {prenda_id}")

        embedding = generar_embedding(url, mostrar_logs=mostrar_logs)

        if embedding is None:
            if mostrar_logs:
                print(f"No se pudo registrar la prenda ID {prenda_id}")
            continue

        prendas_con_embeddings.append({
            "id": prenda_id,
            "url": url,
            "embedding": embedding,
            **referencia["metadata"]
        })

        if mostrar_logs:
            print(f"Prenda ID {prenda_id} registrada correctamente")

    return prendas_con_embeddings


def escanear_prenda(
    imagen_a_escanear,
    imagenes_referencia=None,
    umbral_minimo=0.70,
    mostrar_logs=True,
    margen_top_matches=0.05,
    max_top_matches=None,
    conjunto_de_prendas=None,
    umbral=None
):
    """
    Escanea una imagen JPG directa y la compara contra referencias.

    Contrato principal para Eduardo:
    escanear_prenda(
        imagen_a_escanear=<bytes JPG recibidos por backend>,
        imagenes_referencia=[
            {"id": 1, "url": "https://drive.google.com/uc?id=..."},
            {"id": 2, "url": "https://drive.google.com/file/d/ID/view?usp=drivesdk"}
        ],
        umbral_minimo=90
    )

    Tambien se puede llamar con:
    - umbral_minimo=0.90 o umbral_minimo=90.
    - imagenes_referencia como lista de strings con URLs.
    - imagenes_referencia como lista ya registrada con embeddings.

    Compatibilidad con tu version anterior:
    escanear_prenda(
        imagen_a_escanear=...,
        conjunto_de_prendas=registrar_prendas(...),
        umbral=0.70
    )

    top_matches:
    - Se ordena de mayor a menor score.
    - Por defecto incluye resultados con score >= umbral_minimo - 5 puntos.
      Ejemplo: si el umbral es 70, top_matches muestra scores >= 65.
    - Si quieres que top_matches use exactamente el umbral, manda margen_top_matches=0.
    - Si quieres limitar cantidad, manda max_top_matches=5, 10, etc.
    """

    # Compatibilidad con nombre anterior del parametro.
    if umbral is not None:
        umbral_minimo = umbral

    if imagenes_referencia is None and conjunto_de_prendas is not None:
        imagenes_referencia = conjunto_de_prendas

    if imagenes_referencia is None:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": 0,
            "similitud_porcentaje": 0,
            "best_match": None,
            "top_matches": [],
            "referencias_fallidas": [],
            "mensaje": "No se recibieron imagenes de referencia."
        }

    try:
        umbral_decimal = normalizar_umbral(umbral_minimo, nombre="umbral_minimo")
        margen_decimal = normalizar_umbral(margen_top_matches, nombre="margen_top_matches")
    except ValueError as e:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": 0,
            "similitud_porcentaje": 0,
            "best_match": None,
            "top_matches": [],
            "referencias_fallidas": [],
            "mensaje": str(e)
        }

    umbral_top_matches = max(0, umbral_decimal - margen_decimal)

    # La imagen a escanear debe llegar como JPG/bytes desde el backend.
    # Se mantiene soporte para ruta local o URL solo para pruebas.
    embedding_nuevo = generar_embedding(
        imagen_a_escanear,
        mostrar_logs=mostrar_logs
    )

    if embedding_nuevo is None:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": 0,
            "similitud_porcentaje": 0,
            "best_match": None,
            "top_matches": [],
            "referencias_fallidas": [],
            "umbral_minimo": round(umbral_decimal, 4),
            "umbral_minimo_porcentaje": formatear_porcentaje(umbral_decimal),
            "umbral_top_matches": round(umbral_top_matches, 4),
            "umbral_top_matches_porcentaje": formatear_porcentaje(umbral_top_matches),
            "mensaje": "No se pudo generar el embedding de la imagen enviada."
        }

    resultados = []
    referencias_fallidas = []

    for indice, referencia_original in enumerate(imagenes_referencia):
        referencia = _normalizar_referencia(referencia_original, indice)
        prenda_id = referencia["id"]
        url = referencia["url"]
        embedding_referencia = referencia["embedding"]

        if embedding_referencia is None:
            if url is None:
                referencias_fallidas.append({
                    "id": prenda_id,
                    "url": None,
                    "motivo": "Referencia sin URL ni embedding."
                })
                continue

            embedding_referencia = generar_embedding(
                url,
                mostrar_logs=mostrar_logs
            )

        if embedding_referencia is None:
            referencias_fallidas.append({
                "id": prenda_id,
                "url": url,
                "motivo": "No se pudo leer o procesar la imagen de referencia."
            })
            continue

        score = cosine_similarity(embedding_nuevo, embedding_referencia)
        score_redondeado = round(float(score), 4)

        resultado = {
            "id": prenda_id,
            "url": url,
            "score": score_redondeado,
            "score_porcentaje": formatear_porcentaje(score),
            **referencia["metadata"]
        }

        resultados.append(resultado)

        if mostrar_logs:
            print(f"Comparando con prenda ID {prenda_id}: {score_redondeado}")

    resultados = sorted(
        resultados,
        key=lambda x: x["score"],
        reverse=True
    )

    if not resultados:
        return {
            "detectado": False,
            "prenda_id": None,
            "similitud": 0,
            "similitud_porcentaje": 0,
            "best_match": None,
            "top_matches": [],
            "referencias_fallidas": referencias_fallidas,
            "umbral_minimo": round(umbral_decimal, 4),
            "umbral_minimo_porcentaje": formatear_porcentaje(umbral_decimal),
            "umbral_top_matches": round(umbral_top_matches, 4),
            "umbral_top_matches_porcentaje": formatear_porcentaje(umbral_top_matches),
            "total_referencias": len(imagenes_referencia),
            "referencias_procesadas": 0,
            "mensaje": "No se pudo procesar ninguna imagen de referencia."
        }

    best_match = resultados[0]
    detectado = best_match["score"] >= umbral_decimal

    top_matches = [
        resultado
        for resultado in resultados
        if resultado["score"] >= umbral_top_matches
    ]

    if max_top_matches is not None:
        top_matches = top_matches[:int(max_top_matches)]

    if mostrar_logs:
        print("Mejor similitud:", best_match["score"])
        print("Mejor prenda ID:", best_match["id"])
        print("Umbral minimo:", umbral_decimal)
        print("Umbral top_matches:", umbral_top_matches)

    return {
        "detectado": detectado,
        "prenda_id": best_match["id"] if detectado else None,
        "similitud": best_match["score"],
        "similitud_porcentaje": best_match["score_porcentaje"],
        "best_match": best_match,
        "top_matches": top_matches,
        "referencias_fallidas": referencias_fallidas,
        "umbral_minimo": round(umbral_decimal, 4),
        "umbral_minimo_porcentaje": formatear_porcentaje(umbral_decimal),
        "umbral_top_matches": round(umbral_top_matches, 4),
        "umbral_top_matches_porcentaje": formatear_porcentaje(umbral_top_matches),
        "total_referencias": len(imagenes_referencia),
        "referencias_procesadas": len(resultados),
        "mensaje": (
            "Prenda detectada correctamente."
            if detectado
            else "No se encontro una prenda suficientemente parecida."
        )
    }


def ejecutar_prueba():
    """
    Funcion solo para pruebas locales.

    Esta funcion NO se ejecuta cuando Eduardo importe este archivo.
    Solo se ejecuta cuando corres directamente:

    python main.py

    Nota:
    En produccion, imagen_a_escanear deberia llegar como JPG/bytes desde
    el backend. Aqui se usa URL solo para facilitar pruebas locales.
    """

    prendas_usuario = [
        {
            # gorra bulls 1
            "id": 1,
            "url": "https://drive.google.com/uc?id=1U1MIGL-voE901McRb-ydu9Xm3Sj7flMH"
        },
        {
            # gorra bulls 2
            "id": 2,
            "url": "https://drive.google.com/uc?id=1N0wZwx5Xc1_bj_XrNg0ShGIJ-FU-WDiZ"
        },
        {
            # pantalon azul
            "id": 3,
            "url": "https://drive.google.com/uc?id=1eIckdPm4bDodOw4D1VNvQ31vCggl0Y5x"
        },
        {
            # sueter verde 1
            "id": 4,
            "url": "https://drive.google.com/uc?id=1Hb8dcYTrzQK1OQu8c94B4ZJqGTP4jt_G"
        },
        {
            # camisa vaquera de cuadros
            "id": 5,
            "url": "https://drive.google.com/uc?id=1qetaadJSRr48qaOhkQ8hkfMrRMP0DTMB"
        },
        {
            # camisa lisa gris 3xl
            "id": 6,
            "url": "https://drive.google.com/uc?id=1WFli8HFXq-txJ-WYtr8sxUHGrNEy4Zat"
        },
        {
            # camisa de cuadros mixta
            "id": 7,
            "url": "https://drive.google.com/uc?id=1KpCoanICuOuX7y4HOPI6QYr_31s1Mhhj"
        },
        {
            # camisa negra de botones
            "id": 10,
            "url": "https://drive.google.com/uc?id=1JoncWFE0r-SWbKXs4H8oVYw2_MWChSQa"
        },
        {
            # camisa negra lisa
            "id": 8,
            "url": "https://drive.google.com/uc?id=1NdX3LXPgqBE9PryVl4NjxBMcDI588Y7G"
        },
        {
            # sudadera negra sin flash
            "id": 9,
            "url": "https://drive.google.com/uc?id=1le9IUh_qgu2K56w08Rcmi0y4BhZoIEvu"
        },
        {
            #camisa verde 3xl
            "id": 11,
            "url": "https://drive.google.com/file/d/1IHEuGD4SRLTdupPKwi36vnmf3Q2qQAnq/view?usp=sharing"
        }
    ]

    # Solo para prueba local. En backend real deberia ser bytes JPG.
    imagen_a_escanear = "https://drive.google.com/file/d/1H0y-Vjk1pAx04fszFjcx41-UVQzcXrwW/view?usp=sharing"

    resultado = escanear_prenda(
        imagen_a_escanear=imagen_a_escanear,
        imagenes_referencia=prendas_usuario,
        umbral_minimo=70,
        mostrar_logs=True,
        margen_top_matches=5,
        max_top_matches=None
    )

    print("\nResultado final:")
    print(json.dumps(resultado, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    ejecutar_prueba()
