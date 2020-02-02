import logging

import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers

logger = logging.getLogger("MAIN.MODEL")

class Model(keras.Model):

    def __init__(self, vocab_size, embedding_dim):
        super(Model, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, input_length=1, name="embedding")
        self.reshaper_emb = layers.Reshape((embedding_dim, 1))
        self.dot_product = layers.Dot(axes=1)
        self.reshaper_dot = layers.Reshape((1,))
        self.dense = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        input_target = inputs[0]
        input_context = inputs[1]
        embedded_target = self.embedding(input_target)
        embedded_target = self.reshaper_emb(embedded_target)
        embedded_context = self.embedding(input_context)
        embedded_context = self.reshaper_emb(embedded_context)
        dot_product = self.dot_product([embedded_target, embedded_context])
        dot_product = self.reshaper_dot(dot_product)
        output = self.dense(dot_product)
        return output

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

    # define dot product as similarity measure
    dot_product = layers.dot([embedded_target, embedded_context], axes=1)
    dot_product = layers.Reshape((1,))(dot_product)

    # define sigmoid output layer
    output = layers.Dense(1, activation="sigmoid")(dot_product)

    # create training model
    train_model = keras.Model(inputs=[input_target, input_context], outputs=output)
    logger.debug("created training model")
    train_model.summary()
    train_model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.RMSprop())
    logger.debug("compiled training model")

    # create validation model 
    similarity = layers.dot(inputs=[embedded_target, embedded_context], axes=1, normalize=True)
    val_model = keras.Model(inputs=[input_target, input_context], outputs=similarity)
    logger.debug("created validation model")
    
    return train_model, val_model