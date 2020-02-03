import logging
import sys 

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import *
from preprocessing import * 

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("EVALUATOR")

def load_model(path):
    """Loads a Keras embedding model from path and returns embedding layer."""
    logger.debug(f"loading keras model from path {path}")

    model = keras.models.load_model(path)
    model.summary()
    logger.debug(f"loaded keras model")

    return model

def cos_similarity(word_a, word_b, model):

    # get embedded words
    embedding = model.get_layer(name="embedding")
    word_a_embedded = embedding(word_a)
    word_b_embedded = embedding(word_b)
    
    # normalize and calculate cosine similarity
    word_a_embedded = word_a_embedded / tf.sqrt(tf.reduce_sum(tf.square(word_a_embedded)))
    word_b_embedded = word_b_embedded / tf.sqrt(tf.reduce_sum(tf.square(word_b_embedded)))
    cos_sim = tf.tensordot(word_a_embedded, word_b_embedded, axes=1)

    return cos_sim

def get_most_similar(word_a, k, model):

    # get embedded words
    embedding = model.get_layer(name="embedding")
    word_a_embedded = embedding(word_a)
    

if __name__ == '__main__':

    vocab_size = 10000

    # recreate dictionaries
    input_dir = "input/experiments"
    text_dir = "output/experiments/text"
    words = get_word_tokens(input_dir)
    words = normalization(words, text_dir)
    data, count, word2id, id2word = build_dataset(words, vocab_size)

    loaded_model = load_model("output/experiments/models/final/model_experiments_31_01_2020_173703.h5")

    test_words = ["power", "weak", "mighty", "evil", "family"]
    female_word = "she"
    male_word = "he"
    she_id = word2id[female_word]
    he_id = word2id[male_word]

    for test_word in test_words:
        test_id = word2id[test_word]
        she_sim = cos_similarity(she_id, test_id, loaded_model)
        he_sim = cos_similarity(he_id, test_id, loaded_model)
        logger.debug(f"--- similiarity for: {test_word}")
        logger.debug(f"{female_word}: {she_sim}")
        logger.debug(f"{male_word}: {he_sim}")

