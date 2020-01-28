import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing import *

INPUT_DIR = "./input"
OUTPUT_DIR = "./output"
VOCAB_SIZE = 10000
WINDOW_SIZE = 3
EMBEDDING_DIM = 300
ITERATIONS = 5000
VALIDATION_SIZE = 16
VALIDATION_WINDOW = 100

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
    words_target, words_context = zip(*couples)
    words_target = np.array(words_target, dtype="int32")
    words_context = np.array(words_context, dtype="int32")

    ## CREATE KERAS MODEL ### using functional API

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
    logging.debug("created model")
    logging.debug(model.summary())
    logging.debug("compiling model")
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")

    # validation model 
    similarity = layers.dot(inputs=[embedded_target, embedded_context], axes=1, normalize=True)
    validation_model = keras.Model(inputs=[input_target, input_context], outputs=similarity)
    validation_samples = np.random.choice(VALIDATION_WINDOW, VALIDATION_SIZE, replace=False)

    class SimilarityCallback:
        def run_sim(self):
            for i in range(VALIDATION_SIZE):
                valid_word = id2word[validation_samples[i]]
                top_k = 8  # number of nearest neighbors
                sim = self._get_sim(validation_samples[i])
                nearest = (-sim).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = id2word[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)

        @staticmethod
        def _get_sim(valid_word_idx):
            sim = np.zeros((VOCAB_SIZE,))
            in_arr1 = np.zeros((1,))
            in_arr2 = np.zeros((1,))
            for i in range(VOCAB_SIZE):
                in_arr1[0,] = valid_word_idx
                in_arr2[0,] = i
                out = validation_model.predict_on_batch([in_arr1, in_arr2])
                sim[i] = out
            return sim
    
    sim_cb = SimilarityCallback()

    word_target = np.zeros((1,))
    word_context = np.zeros((1,))
    label = np.zeros((1,))

    # training loop 
    logging.debug("starting training")
    for iteration in range(ITERATIONS):
        # select training data randomly
        index = np.random.randint(0, len(labels)-1)
        word_target[0,] = words_target[index]
        word_context[0,] = words_context[index]
        label[0,] = labels[index]

        loss = model.train_on_batch([word_target, word_context], label)
        if iteration % 100 == 0:
            logging.debug(f"iteration: {iteration} \t loss: {loss}")
        if iteration % 1000 == 0:
            sim_cb.run_sim()