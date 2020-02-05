import os
import sys
import argparse
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator

logger = logging.getLogger("MAIN.TRAIN")

def train(model, id2word, vocab_size, batch_size, epochs, words_target, words_context, labels, chkpts_dir, timestamp, note):
    
    if not os.path.exists(chkpts_dir):
        os.makedirs(chkpts_dir)
        logger.debug(f"created checkpoints models directory {chkpts_dir}")

    # prepare splits
    # train_val_split = 0.2
    full_size = len(labels)
    # train_size = int((1-train_val_split)* full_size)
    # val_size = int(train_val_split * full_size)
    
    # generate full dataset
    dataset = tf.data.Dataset.from_tensor_slices(({"input_1": words_target, "input_2": words_context}, labels))
    dataset = dataset.shuffle(full_size).batch(batch_size)
    
    # split up full dataset 
    # train_dataset = dataset.take(train_size).batch(batch_size)
    # val_dataset = dataset.skip(train_size).batch(batch_size)

    logger.debug("starting training")
    history = model.fit(dataset, epochs=epochs, verbose=1)
    
    logger.debug("finished training")

    return history

def plot(history, path, note, timestamp):
    """Plots loss history.
    
    Args:
        history (History object): records of metrics at epochs
        path (string): where to save the plot
        timestamp (string): when this session started
    """
    rcParams['font.family'] = "Arial" 
    rcParams['xtick.labelsize'] = 11 
    rcParams['ytick.labelsize'] = 11 
    rcParams['axes.labelsize'] = 12 
    rcParams['axes.titlesize'] = 12 
    rcParams['axes.grid'] = True

    logger.debug("plotting loss and accuracy")
    
    loss_name = f"{path}/loss_{timestamp}_{note}.png"
    
    # plot loss
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    plt.plot(history.history["loss"])
    # plt.plot(history.history["val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(0, len(history.history["loss"])-1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.legend(["Train", "Validation"], loc="upper right")
    plt.show()
    fig.savefig(loss_name, dpi=400)
    logger.debug(f"saved loss plot to {loss_name}")