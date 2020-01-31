import os
import datetime
import argparse
import logging

import tensorflow as tf

from preprocessing import *
from train import *

INPUT_DIR = "./input"
LOGS_DIR = "./output/logs"
MODELS_DIR = "./output/models/final"
CHKPTS_DIR = "./output/models/chkpts"
OUTPUT_DIR = "./output/text"

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

    # logger setup
    timestamp = datetime.datetime.now().strftime(format="%d_%m_%Y_%H%M%S")
    log_name = timestamp + ".log"
    log_dir = os.path.join(LOGS_DIR, args.input_subdir) 
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logFormatter = logging.Formatter("%(asctime)s - %(name)s - [%(levelname)s]  %(message)s")
    logger = logging.getLogger("MAIN")

    fileHandler = logging.FileHandler(f"{log_dir}/{log_name}")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    logger.setLevel(logging.DEBUG)
    logger.debug("starting program")

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.debug(f"created final models directory {MODELS_DIR}")

    if not os.path.exists(CHKPTS_DIR):
        os.makedirs(CHKPTS_DIR)
        logger.debug(f"created checkpoints models directory {CHKPTS_DIR}")

    # preprocessing of text
    words = get_word_tokens(INPUT_DIR, args.input_subdir)
    words = normalization(words, OUTPUT_DIR, args.input_subdir)
    data, count, word2id, id2word = build_dataset(words, args.vocab_size)
    words_target, words_context, labels = keras_preprocessing(data, args.vocab_size, args.window_size)

    model, loss_hist = train(words_target, words_context, labels, args.input_subdir, args.embedding_dim,
                  args.iterations, args.vocab_size, args.window_size, CHKPTS_DIR, timestamp)

    # save model
    model_name = f"model_{args.input_subdir}_{timestamp}.h5"
    model.save(os.path.join(MODELS_DIR, model_name))
    logger.debug(f"saved final model {model_name} to {MODELS_DIR}")

    plot_loss(loss_hist, log_dir, timestamp)

if __name__ == '__main__':
    # TODO download more data
    # TODO get logging to work
    # TODO check what exatra intra/inter difference is
    # TODO parallelize preprocessing
    # TODO make model as class
    # TODO test if using model.fit increases scores (increase batch size to > 1)
    main()