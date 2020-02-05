import os
import datetime
import argparse
import logging

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

def main():
    parser = argparse.ArgumentParser("Trains a Word2Vec model with negative sampling")
    parser.add_argument("--input_subdir", dest="input_subdir", required=True,
                        help="The subdirectory in './input' containing training data as '*.txt‘ files. This will also be the name of the output subdirectory")
    parser.add_argument("--embedding_dim", dest="embedding_dim", type=int,
                        help="The embedding dimension of the Word2Vec model (default=300)", default=300)
    parser.add_argument("--vocab_size", dest="vocab_size", type=int,
                        help="The vocabulary size of the Word2Vec model (default=10000)", default=10000)
    parser.add_argument("--window_size", dest="window_size", type=int,
                        help="The number of context words to consider left and right from target word (default=3)", default=3)
    parser.add_argument("--batch_size", dest="batch_size", type=int, 
                        help="The batch size used for training (default=128)", default=128)
    parser.add_argument("--epochs", dest="epochs", type=int, 
                        help="The number of epochs to train the model (default=10)", default=10)
    parser.add_argument("--num_threads", dest="num_threads", type=int,
                        help="The number of threads to run (default=16)", default=16)
    parser.add_argument("--note", dest="note", type=str,
                        help="A note to add to the model and log saving path (default="")", default="")
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
    log_name = timestamp + args.note + ".log"
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

    # preprocessing of text
    words = get_word_tokens(input_dir)
    words = normalization(words, text_dir)
    data, count, word2id, id2word = build_dataset(words, args.vocab_size)
    words_target, words_context, labels = keras_preprocessing(data, args.vocab_size, args.window_size)

    model = build_model(args.vocab_size, args.embedding_dim)
    
    history = train(model,
                    id2word, 
                    args.vocab_size, 
                    args.batch_size, 
                    args.epochs,
                    words_target, 
                    words_context, 
                    labels,
                    chkpts_dir, 
                    timestamp, 
                    args.note)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.debug(f"created final models directory {models_dir}")

    # save model
    model_name = f"model_{timestamp}_{args.note}.h5"
    model.save(os.path.join(models_dir, model_name))
    logger.debug(f"saved final model {model_name} to {models_dir}")

    plot(history, logs_dir, args.note, timestamp)

if __name__ == '__main__':
    # TODO download more data
    # TODO check if preprocessed words already there
    # TODO check if input folder empty
    # TODO check what exatra intra/inter difference is
    # TODO parallelize preprocessing
    main()