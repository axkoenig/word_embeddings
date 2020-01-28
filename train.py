import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing import *

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
VOCAB_SIZE = 1000
WINDOW_SIZE = 3
EMBEDDING_DIM = 300
ITERATIONS = 5000

# TODO consider creating validation batches

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("starting program")

    # preprocessing of text
    words = get_word_tokens(INPUT_DIR)
    words = normalization(words, OUTPUT_DIR)
    data, count, word2id, id2word = build_dataset(words, VOCAB_SIZE)

    # sampling table to produce negative samples in a balanced manner  
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(VOCAB_SIZE)

    # get word pairs (label = 1 for positive samples, 0 for negative samples) 
    couples, labels = tf.keras.preprocessing.sequence.skipgrams(data, VOCAB_SIZE, window_size=WINDOW_SIZE, sampling_table=sampling_table)

    # split tuple into target and context word 
    word_target, word_context = zip(*couples)
    word_target = np.array(word_target, dtype="int32")
    word_context = np.array(word_context, dtype="int32")