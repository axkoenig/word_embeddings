import tensorflow as tf 
import tensorflow.keras as keras

class Model(keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        
    def call(self, inputs):
        