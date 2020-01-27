import logging
import re
import string
import os
from itertools import chain
import collections

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

DATA_DIR = "./data"
OUTPUT_DIR = "./output"
VOCAB_SIZE = 1000

def get_word_tokens(path):
    """Reads all .txt files in given directory and returns tokenized words"""
    
    logging.debug("loading texts")
    text = ""

    # read files from directory
    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r") as f:
                text = " ".join([text, f.read()])

    # tokenize text
    logging.debug("tokenizing texts")
    sentences = nltk.sent_tokenize(text)
    words = [nltk.word_tokenize(sentence) for sentence in sentences]

    # flatten list
    words = list(chain.from_iterable(words))

    return words

def normalization(words):
    """Returns normalized words"""
    logging.debug("normalizing text")

    # write text before processing to disk
    with open(f'{OUTPUT_DIR}/pre.txt', 'w') as f:
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
    with open(f'{OUTPUT_DIR}/post.txt', 'w') as f:
        for word in words:
            f.write("%s\n" % word)

    logging.debug(f"normalizing text done. returning {len(words)} words")
    return words

def build_dataset(words, n_words):
    """Taken from https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/word2vec/word2vec_basic.py"""
    
    logging.debug("building dataset")

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
    
    logging.debug(f"unique words: {len(set(words))}")
    logging.debug(f"most common words: {count[:10]}")
    logging.debug("building dataset done")

    return data, count, word2id, id2word


def main():
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug("starting program")

    words = get_word_tokens(DATA_DIR)
    words = normalization(words)
    data, count, word2id, id2word = build_dataset(words, VOCAB_SIZE)

if __name__ == '__main__':
    main()