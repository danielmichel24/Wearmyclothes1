import tensorflow as tf
import numpy as np
from PIL import Image

modelo = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

def cargar_imagen(ruta):
    try:
        img = Image.open(ruta)

        # Convertir a RGB (esto elimina problemas con PNG, WEBP, etc.)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((224, 224))

        img = np.array(img)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

        return np.expand_dims(img, axis=0)

    except Exception as e:
        print(f"Error cargando imagen {ruta}: {e}")
        return None

def generar_embedding(ruta):
    img = cargar_imagen(ruta)

    if img is None:
        return None

    embedding = modelo.predict(img, verbose=0)
    return embedding[0]