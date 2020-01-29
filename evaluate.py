import numpy as np
import tensorflow as tf
from tensorflow import keras

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
global_logger = logging.getLogger("evaluator")

def load_model(path):
    """Loads a Keras embedding model from path and returns embedding layer."""
    global_logger.debug(f"loading keras model from path '{path}'")

    model = keras.models.load_model(path)
    model.summary()
    embedding = model.layers[2].get_weights()[0]
    
    global_logger.debug(f"loading keras model done")

    return embedding