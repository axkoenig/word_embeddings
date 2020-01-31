import os
import sys
import argparse
import logging

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from matplotlib import rcParams

logger = logging.getLogger("MAIN.TRAIN")

def train(words_target, words_context, labels, input_subdir, embedding_dim, iterations, vocab_size, window_size, chkpts_dir, timestamp):
    
    if not os.path.exists(chkpts_dir):
        os.makedirs(chkpts_dir)
        logger.debug(f"created checkpoints models directory {chkpts_dir}")

    # create input variables and embedding layer
    input_target = keras.Input((1,))
    input_context = keras.Input((1,))
    embedding = layers.Embedding(vocab_size, embedding_dim, input_length=1, name="embedding")

    # embed target and context words, reshape for dot product
    embedded_target = embedding(input_target)
    embedded_target = layers.Reshape((embedding_dim, 1))(embedded_target)
    embedded_context = embedding(input_context)
    embedded_context = layers.Reshape((embedding_dim, 1))(embedded_context)

    # define dot product as similarity measure
    dot_product = layers.dot([embedded_target, embedded_context], axes=1)
    dot_product = layers.Reshape((1,))(dot_product)

    # define sigmoid output layer
    output = layers.Dense(1, activation="sigmoid")(dot_product)

    # create model
    model = keras.Model(inputs=[input_target, input_context], outputs=output)
    logger.debug("created model")
    logger.debug(model.summary())
    logger.debug("compiling model")
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")

    word_target = np.zeros((1,))
    word_context = np.zeros((1,))
    label = np.zeros((1,))

    logger.debug("starting training")
    avg_loss = 0
    loss_hist = []
    log_iter = 2
    chkpt_iter = 10000

    for iteration in range(iterations):
        # select training data randomly
        index = np.random.randint(0, len(labels)-1)
        word_target[0, ] = words_target[index]
        word_context[0, ] = words_context[index]
        label[0, ] = labels[index]

        loss = model.train_on_batch([word_target, word_context], label)
        avg_loss += loss

        if iteration % log_iter == 0:
            if iteration > 0:
                avg_loss /= log_iter
            logger.debug(f"iteration: {iteration} \t avg loss: {avg_loss}")
            loss_hist.append((iteration, avg_loss))
            avg_loss = 0

        if iteration % chkpt_iter == 0:
            logger.debug(f"saving model at iteration {iteration}")
            model_name = f"{iteration}_of_{iterations}_model_{input_subdir}_{timestamp}.h5"
            model.save(os.path.join(chkpts_dir, model_name))
            logger.debug(f"saved model {model_name} to {chkpts_dir}")

    logger.debug("finished training")

    return model, loss_hist

def plot_loss(loss_hist, path, timestamp):
    """Plots loss history.
    
    Args:
        loss_hist (array): array of tuples containing iterations and loss
        path (string): where to save the plot
        timestamp (string): when this session started
    """
    rcParams['font.family'] = "Arial" 
    rcParams['xtick.labelsize'] = 11 
    rcParams['ytick.labelsize'] = 11 
    rcParams['axes.labelsize'] = 12 
    rcParams['axes.titlesize'] = 12 
    rcParams['axes.grid'] = True

    logger.debug("plotting loss")
    
    iterations = [i[0] for i in loss_hist]
    losses = [i[1] for i in loss_hist]
    name = f"{path}/loss_{timestamp}.png"

    fig = plt.figure(constrained_layout=True)
    plt.plot(iterations, losses)
    plt.xlabel("Iterations")
    plt.ylabel("Average Loss")
    plt.xlim(0, iterations[-1])
    fig.savefig(name, dpi=400)

    logger.debug(f"saved loss plot to {name}")