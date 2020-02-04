import logging

import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers

logger = logging.getLogger("MAIN.MODEL")

def build_model(vocab_size, embedding_dim):
    """Creates a model using keras functional API""" 

    # create input variables and embedding layer
    input_target = keras.Input((1,))
    input_context = keras.Input((1,))
    embedding = layers.Embedding(vocab_size, embedding_dim, input_length=1, name="embedding")

    # embed target and context words, reshape for dot product
    embedded_target = embedding(input_target)
    embedded_target = layers.Reshape((embedding_dim, 1))(embedded_target)
    embedded_context = embedding(input_context)
    embedded_context = layers.Reshape((embedding_dim, 1))(embedded_context)

    # define cosine similarity (= normalized dot product)
    cosine_sim = layers.dot([embedded_target, embedded_context], axes=1, normalize=True)
    cosine_sim = layers.Reshape((1,))(cosine_sim)

    # define sigmoid activation layer
    output = layers.Activation(activation="sigmoid")(cosine_sim)

    # create and compile model
    model = keras.Model(inputs=[input_target, input_context], outputs=output)
    logger.debug("created model")
    model.summary()
    model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.RMSprop())
    logger.debug("compiled model")

    return model