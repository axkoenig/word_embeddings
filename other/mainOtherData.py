import os
import datetime
import argparse
import logging
import urllib
import zipfile
import io
import collections

import tensorflow as tf

from model import * 
from preprocessing import *
from train import *

INPUT_DIR = "input"
OUTPUT_DIR = "output"
LOGS_DIR = "logs"
MODELS_DIR = "models/final"
CHKPTS_DIR = "models/chkpts"
TEXT_DIR = "text"

def maybe_download(filename, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


# Read the data into a list of strings.
def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words."""
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def main():
    parser = argparse.ArgumentParser("Trains a Word2Vec model with negative sampling")
    parser.add_argument("--input_subdir", dest="input_subdir", required=True,
                        help="The subdirectory in './input' containing training data as '*.txt‘ files. This will also be the name of the output subdirectory")
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

    # define directories
    input_dir = os.path.join(INPUT_DIR, args.input_subdir)
    models_dir = os.path.join(OUTPUT_DIR, args.input_subdir, MODELS_DIR)
    chkpts_dir = os.path.join(OUTPUT_DIR, args.input_subdir, CHKPTS_DIR)
    logs_dir = os.path.join(OUTPUT_DIR, args.input_subdir, LOGS_DIR)
    text_dir = os.path.join(OUTPUT_DIR, args.input_subdir, TEXT_DIR)

    # logger setup
    timestamp = datetime.datetime.now().strftime(format="%d_%m_%Y_%H%M%S")
    log_name = timestamp + ".log"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logFormatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s]  %(message)s")
    logger = logging.getLogger("MAIN")

    fileHandler = logging.FileHandler(f"{logs_dir}/{log_name}")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.DEBUG)
    logger.debug("starting program")

    url = 'http://mattmahoney.net/dc/'
    filename = maybe_download('text8.zip', url, 31344016)
    words = read_data(filename)
    print(words[:7])
    data, count, word2id, id2word = build_dataset(words, args.vocab_size)
    words_target, words_context, labels = keras_preprocessing(data, args.vocab_size, args.window_size)

    # create model 
    train_model, val_model = build_model(args.vocab_size, args.embedding_dim)

    loss_hist = train(train_model, val_model, id2word, args.vocab_size, words_target, words_context, labels, args.iterations, chkpts_dir, timestamp)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.debug(f"created final models directory {models_dir}")

    # save model
    model_name = f"model_{timestamp}.h5"
    train_model.save(os.path.join(models_dir, model_name))
    logger.debug(f"saved final model {model_name} to {models_dir}")

    plot_loss(loss_hist, logs_dir, timestamp)

if __name__ == '__main__':
    # TODO download more data
    # TODO check if input folder empty
    # TODO check what exatra intra/inter difference is
    # TODO parallelize preprocessing
    # TODO test if using model.fit increases scores (increase batch size to > 1)
    main()