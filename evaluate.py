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

def is_context_word(model, word_a, word_b):
    """Calculates probability that both words appear in context with each 
    other by executing forward pass of model.
    
    Args:
        model (Mode): keras model
        word_a (int): index of first word
        word_b (int): index of second word
    """
    # define inputs
    input_a = np.zeros((1,))
    input_b = np.zeros((1,))
    input_a[0,] = word_a
    input_b[0,] = word_b
    
    # compute forward pass of model
    prediction = model.predict_on_batch([input_a, input_b])
    
    # retrieve value from tensor
    prediction = prediction.numpy()[0][0]
    
    return prediction 

def get_similar_words(model, word, vocab_size, n):
    """Returns n most similar words as indices.
    """
    
    similarities = np.zeros((vocab_size, 1))
    
    # get similarities for each word
    for i in range(vocab_size):
        similarities[i] = is_context_word(model, word, i)
    
    # get indices of n most similar words
    most_similar = (-similarities).argsort()[1 : n+1]

    return most_similar
    
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

def calc_gender_bias(model):
    positive_words = ["good", "power", "mighty"]
    negative_words = ["bad", "evil"]
    
    male_words = ["he", "him", "his"]
    female_words = ["she", "her", "hers"]
    
    # if key error remove word from list 
    
    pass   

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
    test_words = words[:5]
    female_word = "she"
    male_word = "he"
    she_id = word2id[female_word]
    he_id = word2id[male_word]
    my_id = word2id["god"]
    
    print(is_context_word(loaded_model, he_id, my_id))
    print(is_context_word(loaded_model, she_id, my_id))
    
    n = 10
    most_similar = get_similar_words(loaded_model, he_id, vocab_size, n)
    
    print("words most similar to --he--")
    for i in range(n):
        print(most_similar[i])
        import pdb; pdb.set_trace()
        word = id2word[most_similar[i]]
        print(word)
        
    # for test_word in test_words:
    #     test_id = word2id[test_word]
    #     she_sim = cos_similarity(she_id, test_id, loaded_model)
    #     he_sim = cos_similarity(he_id, test_id, loaded_model)
    #     logger.debug(f"--- similiarity for: {test_word}")
    #     logger.debug(f"{female_word}: {she_sim}")
    #     logger.debug(f"{male_word}: {he_sim}")