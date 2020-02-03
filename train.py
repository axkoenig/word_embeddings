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

### params for validation
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    
def train(model, val_model, id2word, vocab_size, words_target, words_context, labels, iterations, chkpts_dir, timestamp, note):
    
    if not os.path.exists(chkpts_dir):
        os.makedirs(chkpts_dir)
        logger.debug(f"created checkpoints models directory {chkpts_dir}")

    word_target = np.zeros((1,))
    word_context = np.zeros((1,))
    label = np.zeros((1,))

    logger.debug("starting training")
    avg_loss = 0
    loss_hist = []
    log_iter = 1000
    chkpt_iter = 10000

    sim_cb = SimilarityCallback()
    
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
            model_name = f"{iteration}_of_{iterations}_model_{note}_{timestamp}.h5"
            model.save(os.path.join(chkpts_dir, model_name))
            logger.debug(f"saved model {model_name} to {chkpts_dir}")
            
            sim_cb.run_sim(id2word, val_model, vocab_size)

    logger.debug("finished training")

    return loss_hist

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

class SimilarityCallback:
    def run_sim(self, id2word, val_model, vocab_size):
        for i in range(valid_size):
            valid_word = id2word[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i], val_model, vocab_size)
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = id2word[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx, val_model, vocab_size):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = val_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
