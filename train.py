import os
import datetime 
import sys
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from preprocessing import *

LOGS_DIR = "./logs"
INPUT_DIR = "./input"
MODELS_DIR = "./output/models/final"
CHKPTS_DIR = "./output/models/chkpts"
OUTPUT_DIR = "./output/text"

def train(input_subdir, embedding_dim, iterations, vocab_size, window_size):
    # preprocessing of text
    words = get_word_tokens(INPUT_DIR, input_subdir)
    words = normalization(words, OUTPUT_DIR, input_subdir)
    data, count, word2id, id2word = build_dataset(words, vocab_size)
    words_target, words_context, labels = keras_preprocessing(data, vocab_size, window_size)

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

    # create model
    model = keras.Model(inputs=[input_target, input_context], outputs=output)
    logger.debug("created model")
    logger.debug(model.summary())
    logger.debug("compiling model")
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")

    word_target = np.zeros((1,))
    word_context = np.zeros((1,))
    label = np.zeros((1,))

    logger.debug("starting training")
    avg_loss = 0
    log_iter = 1000
    chkpt_iter = 10000

    for iteration in range(iterations):
        # select training data randomly
        index = np.random.randint(0, len(labels)-1)
        word_target[0, ] = words_target[index]
        word_context[0, ] = words_context[index]
        label[0, ] = labels[index]

        loss = model.train_on_batch([word_target, word_context], label)
        avg_loss += loss

        if iteration % log_iter == 0:
            if iteration > 0:
                avg_loss /= log_iter
            logger.debug(f"iteration: {iteration} \t avg loss: {avg_loss}")
            avg_loss = 0
        
        if iteration % chkpt_iter == 0:
            logger.debug(f"saving model at iteration {iteration}")
            model_name = f"{iteration}_of_{iterations}_model_{input_subdir}_{timestamp}.h5"
            model.save(os.path.join(CHKPTS_DIR, model_name))
            logger.debug(f"saved model {model_name} to {CHKPTS_DIR}")
    
    logger.debug("finished training")

    return model

if __name__ == '__main__':
    # TODO plot model loss
    # TODO download more data 
    # TODO get logging to work
    # TODO check what exatra intra/inter difference is 
    # TODO parallelize preprocessing 
    # TODO make model as class

    parser = argparse.ArgumentParser("Trains a Word2Vec model with negative sampling")
    parser.add_argument("--input_subdir", dest="input_subdir", required=True,
                        help="The subdirectory in './input' containing training data as '*.txt‘ files")
    parser.add_argument("--embedding_dim", dest="embedding_dim", type=int,
                        help="The embedding dimension of the Word2Vec model (default=300)", default=300)
    parser.add_argument("--iterations", dest="iterations", type=int,
                        help="The number of iterations to train for (default=500000)", default=500000)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int,
                        help="The vocabulary size of the Word2Vec model (default=10000)", default=10000)
    parser.add_argument("--window_size", dest="window_size", type=int,
                        help="The number of context words to consider left and right from target word (default=3)", default=3)
    parser.add_argument("--num_threads", dest="num_threads", type=int,
                        help="The number of threads to run (default=16)", default=16)
    args = parser.parse_args()

    # execute program in multiple threads
    tf.config.threading.set_inter_op_parallelism_threads(args.num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.num_threads)

    # logger setup
    timestamp = datetime.datetime.now().strftime(format="%d_%m_%Y_%H%M%S")
    if not os.path.exists(LOGS_DIR):
        os.mkdir(LOGS_DIR)
        logger.debug(f"created logging directory {LOGS_DIR}")
    logging.basicConfig(filename=os.path.join(LOGS_DIR, timestamp+".log"), level=logging.DEBUG)
    logger = logging.getLogger("trainer")
    logger.debug("starting program")

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.debug(f"created final models directory {MODELS_DIR}")
    
    if not os.path.exists(CHKPTS_DIR):
        os.makedirs(CHKPTS_DIR)
        logger.debug(f"created checkpoints models directory {CHKPTS_DIR}")

    model = train(args.input_subdir, args.embedding_dim, args.iterations, args.vocab_size, args.window_size)

    model_name = f"model_{args.input_subdir}_{timestamp}.h5"
    model.save(os.path.join(MODELS_DIR, model_name))
    logger.debug(f"saved final model {model_name} to {MODELS_DIR}")