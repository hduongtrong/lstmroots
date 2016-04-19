import time
import logging
import sys
import collections

import numpy as np
import tensorflow as tf

try:
    logger
except NameError:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s\t%(message)s",
        filename="/tmp/history.log",
        filemode='a', level=logging.DEBUG,
        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)


def Print(s):
    sys.stdout.write(str(s) + "\n")
    sys.stdout.flush()


def PrintMessage(iteration=-1, **kwargs):
    """
    Helper function to print out process while training
    """
    messages = kwargs
    mess_header = '|'.join(["%10s" % title for title in messages.keys()])
    mess_content = '|'.join(["%10.5f" % value for value in messages.values()])
    if iteration < 0:
        logger.info(mess_header)
    else:
        logger.info(mess_content)


def PrintMessage1(header=True, **kwargs):
    kwargs = collections.OrderedDict(sorted(kwargs.items()))
    headers = []
    messages = []
    for key, value in kwargs.iteritems():
        headers.append("%8s" % key)
        if isinstance(value, int):
            messages.append("%8d" % value)
        else:
            messages.append("%8.4f" % value)
    if header:
        logger.info("|".join(headers))
    else:
        logger.info('|'.join(messages))


class PrintMess():
    def __init__(self, print_twice=True):
        self.time = time.time()
        self.print_twice = print_twice

    def PrintMessage2(self, header=True, **kwargs):
        if self.print_twice:
            PrintMessage1(header=header, **kwargs)
        if not header:
            time_spent = time.time() - self.time
            self.time = time.time()
        else:
            time_spent = 0
        kwargs = collections.OrderedDict(sorted(kwargs.items()))
        headers = ["%8s" % "Seconds"]
        messages = ["%8d" % time_spent]
        for key, value in kwargs.iteritems():
            headers.append("%8s" % key)
            if isinstance(value, int):
                messages.append("%8d" % value)
            else:
                messages.append("%8.4f" % value)
        if header:
            Print("|".join(headers))
        else:
            Print('|'.join(messages))


def multinomial_loss(labels, yhat):
    """
    Function to calculate the multinomial loss given the yhat and labels.
        loss = 1/n * \sum_{i,j} {labels * log(yhat)}
    Parameters
    ----------
        labels: Shape n * k
        yhat:   Shape n * k
    Returns:
    --------
        loss:   Scalar
    """
    # This return a tensor shape (n,)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        yhat, labels, name='xentropy')
    return tf.reduce_mean(cross_entropy, name="xentropy_mean")


def mean_squared_error(labels, yhat):
    """
    Function to calculate the Mean Squared Error given labels and yhat.
        loss = mean (labels - yhat)^2
    Parameters
    ----------
        labels: Tensor of Any shape, but same as yhat
        yhat:   Tensor fo any shape, but same as labels

    """
    return tf.reduce_mean((labels - yhat)**2)


def accuracy_score(labels, yhat):
    correct_prediction = tf.equal(tf.argmax(yhat, 1),
                                  tf.argmax(labels, 1))
    eval_correct = tf.reduce_mean(tf.cast(correct_prediction,
                                  tf.float32))
    return eval_correct


def accuracy_score2(labels, yhat):
    correct_prediction = tf.equal(tf.cast(tf.argmax(yhat, 1), tf.int32),
                                  labels)
    eval_correct = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return eval_correct


def r2_score(labels, yhat):
    labels_mean = tf.reduce_mean(labels)
    return 1 - tf.reduce_mean((yhat - labels)**2) / \
        tf.reduce_mean((labels - labels_mean)**2)


def r2_scores(labels, yhat):
    """ When there are multiple y. This will gives Rsquared for each column of
    y. Or the average R-Squared. Note that this is different from the R-squared
    calculated by squashing the matrix into a column.
    """
    labels_mean = tf.reduce_mean(labels, 0, keep_dims=True)
    mse = tf.reduce_mean((labels - yhat)**2, 0)
    tse = tf.reduce_mean((labels - labels_mean)**2, 0)
    r2 = 1 - mse / tse
    return np.mean(r2)


loss_dict = {'mse': mean_squared_error,
             'ce': multinomial_loss,
             'crossentropy': multinomial_loss,
             'cross_entropy': multinomial_loss,
             'mean_squared_error': mean_squared_error}
score_dict = {'mse': r2_score,
              'ce': accuracy_score,
              'crossentropy': accuracy_score,
              'cross_entropy': accuracy_score,
              'mean_squared_error': r2_score}
