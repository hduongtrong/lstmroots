import os
import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
from tensorflow.python.platform import gfile

from utils import r2_score, mean_squared_error
from utils import logger, Print, PrintMess

logger.setLevel("CRITICAL")
logging.disable("INFO")
tf.app.flags.DEFINE_boolean("print_twice", False, "Work interactively or not?")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning Rate")
tf.app.flags.DEFINE_float("max_grad_norm", 1., "Clipping gradient norm")
tf.app.flags.DEFINE_float("init_scale", .1, "Norm of initial weights")
tf.app.flags.DEFINE_integer("hidden_dim", 100, "hidden size of Neural net")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in neural net")
tf.app.flags.DEFINE_integer("input_dim", 1, "Dimension of the target")
tf.app.flags.DEFINE_integer("output_dim", 1, "Dimension of the target")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch Size for SGD")
tf.app.flags.DEFINE_integer("n_iter", 100000, "Number of Iteration")
tf.app.flags.DEFINE_integer("n_valid", 5000, "Number of obs for valid set")
tf.app.flags.DEFINE_integer("freq_mess", 100, "Print a message every ... iter")
tf.app.flags.DEFINE_integer("seed", 2, "Random Number Seed")
tf.app.flags.DEFINE_string("task", "poly_eval", """Choose the task from
                            poly_eval, poly_der_eval, and poly_div,
                            newton_eval""")
tf.app.flags.DEFINE_string("train_degrees", "5,10,15", "The degrees to train")
tf.app.flags.DEFINE_string("valid_degrees", "20", "The single degree to valid")
tf.app.flags.DEFINE_string("train_dir",
                           "/tmp/",
                           "Directory to save model")
FLAGS = tf.app.flags.FLAGS


class RecursiveRNN():
    def __init__(self, learning_rate, input_dim, hidden_dim, output_dim,
                 num_layers, max_grad_norm, seq_lens):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.max_grad_norm = max_grad_norm
        self.seq_lens = seq_lens
        self.global_step = tf.Variable(0, trainable=False)

        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)

        self.__initialize_tensors()
        self.__construct_graph()

    def __initialize_tensors(self):
        # Defining Place Holder for data
        self.input_coef_0 = []
        self.target = []
        for l in xrange(self.seq_lens[-1]):
            self.input_coef_0.append(tf.placeholder(tf.float32,
                                     [None, self.input_dim],
                                     name="inp{0}".format(l)))
        self.target.append(tf.placeholder(tf.float32, [None, self.output_dim],
                           name="target{0}".format(0)))
        self.target.append(tf.placeholder(tf.float32, [None, self.output_dim],
                           name="target{0}".format(1)))
        self.target.append(tf.placeholder(tf.float32, [None, self.output_dim],
                           name="target{0}".format(2)))

        # Defining Variable for each iteration
        self.updates = []
        self.losses = []
        self.grad_norms = []
        self.r2 = []
        self.yhat = []

    def __construct_graph(self):
        # Parameter for the first LSTM and find one root
        root_w = tf.get_variable("root_w", [self.hidden_dim,
                                 self.output_dim])
        root_b = tf.get_variable("root_b", [self.output_dim])
        # Parameter for the second LSTM that factorize
        fact_w = tf.get_variable("fact_w", [self.hidden_dim,
                                 self.output_dim])
        fact_b = tf.get_variable("fact_b", [self.output_dim])

        for l in self.seq_lens:
            Print("Creating RNN model for sequence length %d" % l)
            if l > self.seq_lens[0]:
                tf.get_variable_scope().reuse_variables()

            lstm = rnn_cell.BasicLSTMCell(self.hidden_dim)
            cell = rnn_cell.MultiRNNCell([lstm] * self.num_layers)

            # 0.0 LSTM1 from Coef0 to Root0
            output_root_0, _ = rnn.rnn(cell, self.input_coef_0[:l],
                                       dtype=tf.float32, scope="root")

            yhat_root_0 = tf.matmul(output_root_0[-1], root_w) + root_b

            # 0.1. LSTM2 from Root0 to Coef1
            input_root_0 = []
            for i in xrange(l):
                input_root_0.append(
                        tf.concat(1, [self.input_coef_0[i], yhat_root_0]))

            output_coef_0, _ = rnn.rnn(cell, input_root_0,
                                       dtype=tf.float32, scope="fact")

            # 1.0 LSTM1 from Coef1 to Root1
            input_coef_1 = []
            for i in xrange(l - 1):
                input_coef_1.append(tf.matmul(output_coef_0[i], fact_w) +
                                    fact_b)

            tf.get_variable_scope().reuse_variables()
            output_root_1, _ = rnn.rnn(cell, input_coef_1,
                                       dtype=np.float32, scope="root")
            yhat_root_1 = tf.matmul(output_root_1[-1], root_w) + root_b

            # 1.1. LSTM2 from Root1 to Coef2
            input_root_1 = []
            for i in xrange(l - 1):
                input_root_1.append(
                        tf.concat(1, [input_coef_1[i], yhat_root_0]))
            output_coef_1, _ = rnn.rnn(cell, input_root_1,
                                       dtype=tf.float32, scope="fact")

            # 2.0. LSTM1 from Coef2 to Root 2
            input_coef_2 = []
            for i in xrange(l - 2):
                input_coef_2.append(tf.matmul(output_coef_1[i], fact_w) +
                                    fact_b)
            output_root_2, _ = rnn.rnn(cell, input_coef_2,
                                       dtype=np.float32, scope="root")
            yhat_root_2 = tf.matmul(output_root_1[-1], root_w) + root_b
            self.yhat.append(tf.concat(1, [yhat_root_0, yhat_root_1,
                                           yhat_root_2]))

            loss = .55 * mean_squared_error(self.target[0], yhat_root_0) + \
                .27 * mean_squared_error(self.target[1], yhat_root_1) + \
                .18 * mean_squared_error(self.target[2], yhat_root_2)
            r2 = np.min([r2_score(self.target[0], yhat_root_0),
                        r2_score(self.target[1], yhat_root_1),
                        r2_score(self.target[2], yhat_root_2)])

            self.losses.append(loss)
            self.r2.append(r2)
            self.params = tf.trainable_variables()
            grads = tf.gradients(loss, self.params)
            grads, norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.grad_norms.append(norm)
            self.updates.append(tf.train.AdamOptimizer(self.learning_rate,
                                epsilon=1e-4).apply_gradients(
                                    zip(grads, self.params),
                                    global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, sess, inputs, target, is_train=True):
        """
        Taking one step of Stochastic Gradient Descent
        """
        seq_len = len(inputs)
        index = self.seq_lens.index(seq_len)
        feed_in = {}
        feed_out = []
        feed_out.append(self.losses[index])
        feed_out.append(self.r2[index])
        feed_out.append(self.grad_norms[index])
        if is_train:
            feed_out.append(self.updates[index])
        for l in xrange(seq_len):
            feed_in[self.input_coef_0[l].name] = inputs[l]
        feed_in[self.target[0].name] = target[0]
        feed_in[self.target[1].name] = target[1]
        feed_in[self.target[2].name] = target[2]
        res = sess.run(feed_out, feed_in)
        return res

    def predict(self, sess, inputs):
        seq_len = len(inputs)
        index = self.seq_lens.index(seq_len)
        feed_in = {}
        feed_out = []
        feed_out.append(self.yhat[index])
        for l in xrange(seq_len):
            feed_in[self.input_coef_0[l].name] = inputs[l]
        res = sess.run(feed_out, feed_in)
        return res


def GetPolyEvalData(degrees, batch_size=128):
    """ Generate one batch of data
    """
    assert type(degrees) is list
    degree = np.random.choice(degrees)
    X = np.zeros((degree, batch_size, 1), dtype=np.float32)
    Y = np.zeros((3, batch_size, 1), dtype=np.float32)
    for i in xrange(batch_size):
        roots = np.random.uniform(low=-1, high=1, size=degree - 1)
        roots.sort()
        coefs_0_n = np.polynomial.polynomial.polyfromroots(roots)
        Y[0, i, 0] = roots[0]
        Y[1, i, 0] = roots[1]
        Y[2, i, 0] = roots[2]
        X[:, i, 0] = coefs_0_n[::-1]
    return list(X), list(Y)


def TestGetPolyEvalData(degree):
    X, Y = GetPolyEvalData([degree + 1], batch_size=1)
    # Convert X into batch_size * degree * 2
    X = np.transpose(np.array(X), (1, 0, 2))[0]
    coef_n_0 = X[:, 0]
    root = Y[0][0, 0]
    a = coef_n_0 * root ** np.arange(degree + 1)[::-1]
    a = np.sum(a)
    print(a)
    assert(np.abs(a) < 1e-6)


def GetPolyFieldRoot(degrees, field=5, batch_size=128):
    assert type(degrees) is list
    degree = np.random.choice(degrees)
    X = np.zeros(degree, batch_size, field)
    Y = np.zeros(degree, batch_size, field)
    return X, Y


def create_model(sess):
    seq_lens_train = [int(i) for i in FLAGS.train_degrees.split(",")]
    seq_lens_valid = [int(i) for i in FLAGS.valid_degrees.split(",")]
    seq_lens = seq_lens_train + seq_lens_valid
    Print("%20s: %s" % ("seed", FLAGS.seed))
    for key, value in FLAGS.__dict__['__flags'].iteritems():
        Print("%20s: %s" % (key, str(value)))
    model = RecursiveRNN(
                learning_rate=FLAGS.learning_rate,
                input_dim=FLAGS.input_dim,
                hidden_dim=FLAGS.hidden_dim,
                output_dim=FLAGS.output_dim,
                num_layers=FLAGS.num_layers,
                max_grad_norm=FLAGS.max_grad_norm,
                seq_lens=seq_lens)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameter from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Create momdel with fresh parameters.")
        sess.run(tf.initialize_all_variables())
        np.random.seed(FLAGS.seed)
    return model, seq_lens_train, seq_lens_valid


def train():
    Print("".join(["#"]*80))
    sess = tf.Session()
    print("Creating %d layers of %d units." % (FLAGS.num_layers,
                                               FLAGS.hidden_dim))
    model, seq_lens_train, seq_lens_valid = create_model(sess)
    logger2 = PrintMess(FLAGS.print_twice)
    valid_set = []
    for l in seq_lens_valid:
        valid_set.append(GetPolyEvalData([l], FLAGS.n_valid))

    # Get a batch of data, and make a step
    for i in xrange(FLAGS.n_iter):
        if i % (40 * FLAGS.freq_mess) == 0:
            r2_valid = {}
            for l in seq_lens_valid:
                r2_valid["Deg%d" % l] = 0
            logger2.PrintMessage2(header=True, Iter=0, **r2_valid)

        X, Y = GetPolyEvalData(seq_lens_train, FLAGS.batch_size)
        res = model.step(sess, X, Y, is_train=True)

        # Once in a while, we save checkpoint, print statistics, and run evals
        if i % FLAGS.freq_mess == FLAGS.freq_mess - 1:
            checkpoint_path = os.path.join(
                FLAGS.train_dir, "root%s.ckpt" % FLAGS.train_degrees)
            model.saver.save(sess, checkpoint_path,
                             global_step=model.global_step)
            r2_valid = {}
            for l, deg in enumerate(seq_lens_valid):
                res = model.step(sess, valid_set[l][0], valid_set[l][1],
                                 is_train=False)
                r2_valid["Deg%d" % deg] = res[1]

            logger2.PrintMessage2(header=False,
                                  Iter=int(sess.run(model.global_step)),
                                  **r2_valid)


def valid():
    sess = tf.Session()
    print("Creating %d layers of %d units." % (FLAGS.num_layers,
                                               FLAGS.hidden_dim))
    model, seq_lens_train, seq_lens_valid = create_model(sess)
    for l in seq_lens_valid:
        X, Y = GetPolyEvalData([l], FLAGS.n_valid)
        model.step(sess, X, Y, is_train=False)


if __name__ == "__main__":
    train()
