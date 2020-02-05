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
    
    # word_target = np.zeros((1,))
    # word_context = np.zeros((1,))
    # label = np.zeros((1,))

    # avg_loss = 0
    # loss_hist = []
    # log_iter = 1000
    # chkpt_iter = 10000

    # sim_cb = SimilarityCallback()
    
    # for iteration in range(iterations):
    #     # select training data randomly
    #     index = np.random.randint(0, len(labels)-1)
    #     word_target[0, ] = words_target[index]
    #     word_context[0, ] = words_context[index]
    #     label[0, ] = labels[index]

    #     loss = model.train_on_batch([word_target, word_context], label)
    #     avg_loss += loss

    #     if iteration % log_iter == 0:
    #         if iteration > 0:
    #             avg_loss /= log_iter
    #         logger.debug(f"iteration: {iteration} \t avg loss: {avg_loss}")
    #         loss_hist.append((iteration, avg_loss))
    #         avg_loss = 0

    #     if iteration % chkpt_iter == 0:
    #         logger.debug(f"saving model at iteration {iteration}")
    #         model_name = f"{iteration}_of_{iterations}_model_{note}_{timestamp}.h5"
    #         model.save(os.path.join(chkpts_dir, model_name))
    #         logger.debug(f"saved model {model_name} to {chkpts_dir}")
            
    #         sim_cb.run_sim(id2word, val_model, vocab_size)

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