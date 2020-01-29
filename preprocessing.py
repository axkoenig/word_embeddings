import logging
import re
import string
import os
from itertools import chain
import collections
import sys

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow import keras

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
global_logger = logging.getLogger("preprocessor")

def get_word_tokens(input_path):
    """Reads all .txt files in given directory and returns tokenized words"""
    
    global_logger.debug("loading texts")
    text = ""

    # read files from directory
    for file in os.listdir(input_path):
        if file.endswith(".txt"):
            with open(os.path.join(input_path, file), "r") as f:
                text = " ".join([text, f.read()])

    # tokenize text
    global_logger.debug("tokenizing texts")
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]

    # flatten list
    words = list(chain.from_iterable(words))

    return words

def normalization(words, output_path):
    """Returns normalized words"""
    global_logger.debug("normalizing text")

    # write text before processing to disk
    with open(f'{output_path}/pre.txt', 'w') as f:
        for word in words:
            f.write("%s\n" % word)

    # remove unwanted word tokens such as punctuation or numbers, keep apostrophes
    pattern = "[^A-Za-z']+"
    words = list(filter(None, [re.sub(pattern, "", word) for word in words]))

    # case conversion
    words = [word.lower() for word in words]

    # lemmatization 
    wnl = WordNetLemmatizer()
    words = [wnl.lemmatize(word) for word in words]

    # TODO consider cleaning 's apostrophes

    # write text after processing to disk
    with open(f'{output_path}/post.txt', 'w') as f:
        for word in words:
            f.write("%s\n" % word)

    global_logger.debug(f"normalizing text done. returning {len(words)} words")
    return words

def build_dataset(words, n_words):
    """Taken from https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/word2vec/word2vec_basic.py"""
    
    global_logger.debug("building dataset")

    # get n_words most common words and replace rest with "UNK" token
    count = [["UNK", -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))

    # create dictionary of unique ids for each word
    word2id = dict()
    for word, _ in count:
        word2id[word] = len(word2id)
    
    # create list of ids from word list
    data = list()
    unk_count = 0
    for word in words:
        # check if word is known
        if word in word2id:
            index = word2id[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    
    count[0][1] = unk_count

    # create reverse dictionary 
    id2word = dict(zip(word2id.values(), word2id.keys()))
    
    global_logger.debug(f"unique words: {len(set(words))}")
    global_logger.debug(f"most common words: {count[:10]}")
    global_logger.debug("building dataset done")

    return data, count, word2id, id2word

def keras_preprocessing(data, vocab_size, window_size):
    
    # sampling table to produce negative samples in a balanced manner
    sampling_table = keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # get word pairs (label = 1 for positive samples, 0 for negative samples)
    couples, labels = keras.preprocessing.sequence.skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)

    # split tuple into target and context word
    words_target, words_context = zip(*couples)
    words_target = np.array(words_target, dtype="int32")
    words_context = np.array(words_context, dtype="int32")

    return words_target, words_context, labels