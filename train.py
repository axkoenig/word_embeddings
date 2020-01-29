import os
import datetime 
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocessing import *

LOGS_DIR = "./logs"
INPUT_DIR = "./input"
MODELS_DIR = "./output/models"
OUTPUT_DIR = "./output"
VOCAB_SIZE = 10
WINDOW_SIZE = 3
EMBEDDING_DIM = 300
ITERATIONS = 5
NUM_THREADS=16

def train():
    # preprocessing of text
    words = get_word_tokens(INPUT_DIR)
    words = normalization(words, OUTPUT_DIR)
    data, count, word2id, id2word = build_dataset(words, VOCAB_SIZE)
    words_target, words_context, labels = keras_preprocessing(data, VOCAB_SIZE, WINDOW_SIZE)

    # create input variables and embedding layer
    input_target = keras.Input((1,))
    input_context = keras.Input((1,))
    embedding = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=1, name="embedding")

    # embed target and context words, reshape for dot product
    embedded_target = embedding(input_target)
    embedded_target = layers.Reshape((EMBEDDING_DIM, 1))(embedded_target)
    embedded_context = embedding(input_context)
    embedded_context = layers.Reshape((EMBEDDING_DIM, 1))(embedded_context)

    # define dot product as similarity measure
    dot_product = layers.dot([embedded_target, embedded_context], axes=1)
    dot_product = layers.Reshape((1,))(dot_product)

    # define sigmoid output layer
    output = layers.Dense(1, activation="sigmoid")(dot_product)

    # create model
    model = keras.Model(inputs=[input_target, input_context], outputs=output)
    logger.debug("created model")
    logger.debug(model.summary())
    logger.debug("compiling model")
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")

    word_target = np.zeros((1,))
    word_context = np.zeros((1,))
    label = np.zeros((1,))

    # training loop
    logger.debug("starting training")
    for iteration in range(ITERATIONS):
        # select training data randomly
        index = np.random.randint(0, len(labels)-1)
        word_target[0, ] = words_target[index]
        word_context[0, ] = words_context[index]
        label[0, ] = labels[index]

        loss = model.train_on_batch([word_target, word_context], label)
        if iteration % 100 == 0:
            logger.debug(f"iteration: {iteration} \t loss: {loss}")
    
    logger.debug("finished training")

    return model

if __name__ == '__main__':
    # TODO give model name of input dir

    # execute program in multiple threads
    tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
    tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

    # logger setup
    timestamp = datetime.datetime.now().strftime(format="%d-%m-%Y-%H%M%S")
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)
    logging.basicConfig(filename=os.path.join(LOGS_DIR, timestamp+".log"), level=logging.DEBUG)
    logger = logging.getLogger("trainer")
    logger.debug("starting program")

    model = train()

    if not os.path.exists(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    model.save(os.path.join(MODELS_DIR, "model.h5"))
    logger.debug(f"saved model to {MODELS_DIR}")