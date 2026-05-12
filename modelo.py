import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO


modelo = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(224, 224, 3)
)


def descargar_imagen_desde_url(url):
    """
    Descarga una imagen desde una URL.

    Funciona con:
    - URLs directas de imagen
    - URLs de Google Drive tipo:
      https://drive.google.com/uc?id=...
      https://drive.google.com/uc?export=download&id=...
    """

    try:
        session = requests.Session()

        response = session.get(url, timeout=15)

        if response.status_code != 200:
            print(f"Error descargando imagen. Código HTTP: {response.status_code}")
            return None

        content_type = response.headers.get("Content-Type", "")
        print(f"Content-Type recibido: {content_type}")

        # Si ya recibimos una imagen real, regresamos el contenido
        if "image" in content_type:
            return response.content

        # Google Drive a veces manda una página intermedia con cookie de confirmación
        confirm_token = None

        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                confirm_token = value
                break

        if confirm_token:
            separador = "&" if "?" in url else "?"
            url_confirmada = f"{url}{separador}confirm={confirm_token}"

            response = session.get(url_confirmada, timeout=15)

            if response.status_code != 200:
                print(f"Error descargando imagen confirmada. Código HTTP: {response.status_code}")
                return None

            content_type = response.headers.get("Content-Type", "")
            print(f"Content-Type después de confirmación: {content_type}")

            if "image" in content_type:
                return response.content

        # Si llegó aquí, probablemente Google Drive devolvió HTML
        print("El link no devolvió una imagen real.")
        print("Primeros caracteres recibidos:")
        print(response.text[:300])

        return None

    except Exception as e:
        print(f"Error descargando imagen desde URL: {e}")
        return None


def cargar_imagen(origen_imagen):
    """
    Carga una imagen desde diferentes fuentes:

    1. URL:
       "https://drive.google.com/uc?id=..."

    2. Ruta local:
       "imagenes_prueba/playera.jpg"

    3. Bytes:
       contenido binario de una imagen recibida desde cámara/backend
    """

    try:
        # CASO 1: URL o ruta local
        if isinstance(origen_imagen, str):

            if origen_imagen.startswith("http://") or origen_imagen.startswith("https://"):
                contenido = descargar_imagen_desde_url(origen_imagen)

                if contenido is None:
                    return None

                img = Image.open(BytesIO(contenido))

            else:
                img = Image.open(origen_imagen)

        # CASO 2: bytes
        elif isinstance(origen_imagen, bytes):
            img = Image.open(BytesIO(origen_imagen))

        # CASO 3: tipo no soportado
        else:
            print(f"Tipo de imagen no soportado: {type(origen_imagen)}")
            return None

        # Convertir a RGB para evitar problemas con PNG, WEBP, etc.
        if img.mode != "RGB":
            img = img.convert("RGB")

        # MobileNetV2 requiere imágenes de 224x224
        img = img.resize((224, 224))

        # Convertir imagen a arreglo numérico
        img = np.array(img)

        # Preprocesamiento requerido por MobileNetV2
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        # Agregar dimensión de batch: (224, 224, 3) -> (1, 224, 224, 3)
        return np.expand_dims(img, axis=0)

    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return None


def generar_embedding(origen_imagen):
    """
    Genera un embedding a partir de una imagen.

    La imagen puede venir como:
    - URL
    - ruta local
    - bytes
    """

    img = cargar_imagen(origen_imagen)

    if img is None:
        return None

    embedding = modelo.predict(img, verbose=0)

    return embedding[0]