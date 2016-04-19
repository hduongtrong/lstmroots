import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn

from utils import r2_score, mean_squared_error, logger, PrintMess
from utils import Print, r2_scores

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning Rate")
tf.app.flags.DEFINE_float("max_grad_norm", 1., "Clipping gradient norm")
tf.app.flags.DEFINE_float("init_scale", .1, "Norm of initial weights")
tf.app.flags.DEFINE_integer("hidden_dim", 100, "hidden size of Neural net")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in neural net")
tf.app.flags.DEFINE_integer("input_dim", 2, "Dimension of the target")
tf.app.flags.DEFINE_integer("output_dim", 1, "Dimension of the target")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch Size for SGD")
tf.app.flags.DEFINE_integer("n_iter", 100000, "Number of Iteration")
tf.app.flags.DEFINE_integer("n_valid", 5000, "Number of obs for valid set")
tf.app.flags.DEFINE_integer("freq_mess", 100, "Print a message every ... iter")
tf.app.flags.DEFINE_integer("seed", 1, "Random Number Seed")
tf.app.flags.DEFINE_string("task", "poly_eval", "Choose the task from " +
                           "poly_eval, poly_der_eval, and poly_div, " +
                           "newton_eval")
tf.app.flags.DEFINE_string("train_degrees", "5,10,15", "The degrees to train")
tf.app.flags.DEFINE_string("valid_degrees", "20", "The single degree to test")
FLAGS = tf.app.flags.FLAGS


class RNN():
    def __init__(self, learning_rate, input_dim, hidden_dim, output_dim,
                 num_layers, max_grad_norm, seq_lens, is_seq_output=False):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.max_grad_norm = max_grad_norm
        self.seq_lens = seq_lens
        self.is_seq_output = is_seq_output

        np.random.seed(FLAGS.seed)
        tf.set_random_seed(FLAGS.seed)
        self.inputs = []
        if self.is_seq_output:
            self.target = []
        else:
            self.target = tf.placeholder(tf.float32, [None, self.output_dim],
                                         name="target")
        for l in xrange(self.seq_lens[-1]):
            self.inputs.append(
                tf.placeholder(
                    tf.float32,
                    [None, self.input_dim], name="inp{0}".format(l)))
            if self.is_seq_output:
                self.target.append(tf.placeholder(tf.float32,
                                   [None, self.output_dim],
                                   name="tar{0}".format(l)))

        self.updates = []
        self.losses = []
        self.grad_norms = []
        self.r2 = []
        softmax_w = tf.get_variable("softmax_w", [self.hidden_dim,
                                    self.output_dim])
        softmax_b = tf.get_variable("softmax_b", [self.output_dim])

        for l in self.seq_lens:
            logger.info("Creating RNN model for sequence length %d", l)
            if l > self.seq_lens[0]:
                tf.get_variable_scope().reuse_variables()

            lstm_cell = rnn_cell.BasicLSTMCell(self.hidden_dim)
            # lstm_cell = rnn_cell.BasicRNNCell(self.hidden_dim)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            output, state = rnn.rnn(cell, self.inputs[:l], dtype=tf.float32)

            if self.is_seq_output:
                loss_list = []
                r2_list = []
                for out, tar in zip(output, self.target[:l]):
                    yhat = tf.matmul(out, softmax_w) + softmax_b
                    loss_list.append(mean_squared_error(tar, yhat))
                    r2_list.append(r2_score(tar, yhat))
                loss = tf.python.math_ops.add_n(loss_list) / l
                r2 = tf.python.math_ops.add_n(r2_list) / l
            else:
                yhat = tf.matmul(output[-1], softmax_w) + softmax_b
                loss = mean_squared_error(self.target, yhat)
                if self.output_dim == 1:
                    r2 = r2_score(self.target, yhat)
                else:
                    r2 = r2_scores(self.target, yhat)
            self.losses.append(loss)
            self.r2.append(r2)
            params = tf.trainable_variables()
            grads = tf.gradients(loss, params)
            grads, norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self.grad_norms.append(norm)
            self.updates.append(
                tf.train.AdamOptimizer(
                    self.learning_rate,
                    epsilon=1e-4).apply_gradients(zip(grads, params)))

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
            feed_in[self.inputs[l].name] = inputs[l]
            if self.is_seq_output:
                feed_in[self.target[l].name] = target[l]
        if not self.is_seq_output:
            feed_in[self.target.name] = target
        res = sess.run(feed_out, feed_in)
        return res


def GetPolyEvalData(degrees, batch_size=128):
    """ Generate one batch of data
    """
    assert type(degrees) is list
    if FLAGS.task == "root_eval":
        return GetSmallestRootData(degrees, batch_size)
    degree = np.random.choice(degrees)
    X = np.zeros((degree, batch_size, 2), dtype=np.float32)
    if FLAGS.task.endswith("eval"):
        Y = np.zeros((batch_size, 1), dtype=np.float32)
    else:
        Y = np.zeros((degree, batch_size, 1), dtype=np.float32)
    for i in xrange(batch_size):
        roots = np.random.uniform(low=-1, high=1, size=degree - 1)
        roots.sort()
        coefs_0_n = np.polynomial.polynomial.polyfromroots(roots)
        f = np.poly1d(roots, True)
        coefs_0_n = np.random.uniform(low=-1, high=1, size=degree)
        f = np.poly1d(coefs_0_n[::-1])
        a = np.random.uniform(low=-1, high=1)
        if FLAGS.task == "poly_eval":
            Y[i, 0] = f(a)
        elif FLAGS.task == "poly_der_eval":
            Y[i, 0] = np.polyder(f)(a)
        elif FLAGS.task == "poly_div":
            Y[:, i, 0] = np.concatenate(np.polydiv(f, [1, -a]))
        elif FLAGS.task == "newton_eval":
            Y[i, 0] = a - f(a) / np.polyder(f)(a)
        else:
            raise ValueError("Task must be either poly_eval, poly_div, " +
                             "poly_der_eval, root_eval, or newton_eval")
        X[:, i, 0] = coefs_0_n[::-1]
        X[:, i, 1] = a
    if FLAGS.task.endswith("eval"):
        return list(X), Y
    else:
        return list(X), list(Y)


def GetSmallestRootData(degrees, batch_size):
    assert type(degrees) is list
    degree = np.random.choice(degrees)
    X = np.zeros((degree, batch_size, 1), dtype=np.float32)
    Y = np.zeros((batch_size, 1), dtype=np.float32)
    for i in xrange(batch_size):
        roots = np.random.uniform(low=-1, high=1, size=degree - 1)
        roots.sort()
        coefs_0_n = np.polynomial.polynomial.polyfromroots(roots)
        X[:, i, 0] = coefs_0_n[::-1]
        Y[i, 0] = roots[0]
    return list(X), Y


def TestGetPolyEvalData(degree):
    X, Y = GetPolyEvalData([degree + 1], batch_size=1)
    if FLAGS.task == "poly_eval":
        # Convert X into batch_size * degree * 2
        X = np.transpose(np.array(X), (1, 0, 2))[0]
        coef_n_0 = X[:, 0]
        a = X[0, 1]
        y_true = np.sum(coef_n_0 * (a ** np.arange(degree + 1)[::-1]))
        print(y_true, Y[0, 0])
        assert(np.isclose(y_true, Y[0, 0]))


if __name__ == "__main__":
    Print("\n\n")
    Print("".join(["#"]*80))
    Print("\n\n")
    seq_lens_train = [int(i) for i in FLAGS.train_degrees.split(",")]
    seq_lens_valid = [int(i) for i in FLAGS.valid_degrees.split(",")]
    seq_lens = list(set(seq_lens_train).union(seq_lens_valid))
    seq_lens.sort()
    Print("%20s: %s" % ("seed", FLAGS.seed))
    for key, value in FLAGS.__dict__['__flags'].iteritems():
        Print("%20s: %s" % (key, str(value)))
    np.random.seed(FLAGS.seed)

    valid_data = [GetPolyEvalData([l], FLAGS.n_valid) for l in seq_lens_valid]
    if FLAGS.task.endswith("eval"):
        is_seq_output = 0
    else:
        is_seq_output = 1
    model = RNN(learning_rate=FLAGS.learning_rate,
                input_dim=FLAGS.input_dim,
                hidden_dim=FLAGS.hidden_dim,
                output_dim=FLAGS.output_dim,
                num_layers=FLAGS.num_layers,
                max_grad_norm=FLAGS.max_grad_norm,
                seq_lens=seq_lens,
                is_seq_output=is_seq_output)

    sess = tf.Session()
    initializer = tf.random_uniform_initializer(
            -FLAGS.init_scale, FLAGS.init_scale)
    sess.run(tf.initialize_all_variables())
    logger2 = PrintMess(print_twice=False)
    r2_dict = {}
    for i in xrange(FLAGS.n_iter):
        if i % (40 * FLAGS.freq_mess) == 0:
            for input_valid, _ in valid_data:
                r2_dict[str(len(input_valid))] = 0
            logger2.PrintMessage2(header=True, Iter=0, **r2_dict)

        X, Y = GetPolyEvalData(seq_lens_train, FLAGS.batch_size)
        res = model.step(sess, X, Y, is_train=True)

        if i % FLAGS.freq_mess == 0:
            r2_dict = {}
            for input_valid, output_valid in valid_data:
                res = model.step(sess, input_valid, output_valid,
                                 is_train=False)
                r2_dict[str(len(input_valid))] = res[1]
            logger2.PrintMessage2(header=False, Iter=i / FLAGS.freq_mess,
                                  **r2_dict)
