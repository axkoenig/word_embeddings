import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras import layers

class Model(keras.Model):

    def __init__(self, vocab_size, embedding_dim):
        super(Model, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, input_length=1, name="embedding")
        self.reshaper_emb = layers.Reshape((embedding_dim, 1))
        self.dot_product = layers.Dot(axes=1)
        self.reshaper_dot = layers.Reshape((1,))
        self.dense = layers.Dense(1, activation="sigmoid")

    def call(self, input_target, input_context):
        embedded_target = self.embedding(input_target)
        embedded_target = self.reshaper_emb(embedded_target)
        embedded_context = self.embedding(input_context)
        embedded_context = self.reshaper_emb(embedded_context)
        dot_product = self.dot_product([embedded_target, embedded_context])
        dot_product = self.reshaper_dot(dot_product)
        output = self.dense(dot_product)
        return output 
        
model = Model(vocab_size, embedding_dim)
logger.debug("created model")
model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.RMSprop())
logger.debug("compiled model")