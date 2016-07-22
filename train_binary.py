import tensorflow as tf
import numpy as np
import sys, os, time, itertools
from util import *

class CarryCell(tf.nn.rnn_cell.RNNCell):
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or 'CarryCell'):
            abc = tf.concat(1, [inputs, state])
            hidden = tf.contrib.layers.fully_connected(
                inputs=abc, num_outputs=8, activation_fn=tf.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros, scope='hidden'
            )
            output = tf.contrib.layers.fully_connected(
                inputs=hidden, num_outputs=2, activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros, scope='output'
            )
            carry = tf.contrib.layers.fully_connected(
                inputs=hidden, num_outputs=1, activation_fn=tf.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros, scope='carry'
            )

            return output, carry

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 2


def main():
    np.random.seed(123)
    tf.set_random_seed(1234)

    # alphabet
    alphabet = '01'
    n_symbols = len(alphabet)
    encode = lambda y: map(lambda x: float(x==y), alphabet)

    # inputs
    batch_size = 128
    max_t = 25
    max_input_t = max_t
    max_output_t = max_t

    # build model
    exp1_ph = tf.placeholder('float', (batch_size, max_input_t, n_symbols))
    exp2_ph = tf.placeholder('float', (batch_size, max_input_t, n_symbols))
    res_ph = tf.placeholder('float', (batch_size, max_output_t, n_symbols))
    abs_loss_ph = tf.placeholder('float', name='absolute_loss')

    # n_hidden_units = 8
    # encoder = tf.nn.rnn_cell.BasicRNNCell(n_hidden_units)
    encoder = CarryCell()

    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder, tf.concat(2, [exp1_ph, exp2_ph]), [max_t] * batch_size, scope='encoder', dtype='float')
    # no input during decoding
    # decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder, tf.zeros((batch_size, max_output_t, n_symbols)), output_seq_lens_ph, initial_state=encoder_final_state, scope='decoder')
    prediction = encoder_outputs

    # loss with indicators
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.reshape(prediction, (-1, n_symbols)), tf.reshape(res_ph, (-1, n_symbols))))

    # optimization
    n_max_iter = int(sys.argv[2])
    n_eval_interval = 1
    n_train_step = 10**3
    global_step = tf.Variable(0, trainable=False, name='global_step')
    initial_learning_rate = 0.01
    decay_steps = 256
    decay_rate = 0.9
    momentum = 0.5
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # summary
    tf.scalar_summary('loss', loss)
    tf.scalar_summary('abs_loss', abs_loss_ph)
    tf.scalar_summary('perplexity', tf.exp(loss))
    tf.scalar_summary('learning_rate', learning_rate)
    # tf.histogram_summary('gradient', optimizer.compute_gradients(loss, var_list=[w12])[0][0])
    summary_op = tf.merge_all_summaries()

    # saver
    checkpoint_dir = 'models/' + sys.argv[3]
    # checkpoint_dir = 'models/rnn-bin-11'
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

    # read data
    data = []
    with open(sys.argv[1]) as fi:
        for l in fi:
            x, y = l.strip().split()
            # parse x
            x1, x2 = x[2:-1].split(',')
            x1, x2 = x1[::-1], x2[::-1]
            y = y[::-1]
            data.append((x1, x2, y))

    with tf.Session() as sess:
        restore_vars(saver, sess, checkpoint_dir)
        writer = tf.train.SummaryWriter('tf-log/%s-%d' % (sys.argv[3], time.time()), sess.graph)

        # test the model with all possible inputs
        with tf.variable_scope('encoder', reuse=True):
            for a in [[0., 1.], [1., 0.]]:
                for b in [[0., 1.], [1., 0.]]:
                    for c in [[0.], [1.]]:
                        o, s = encoder([a + b], [c])
                        print a, b, c, o.eval(), s.eval()

        step = 0
        batch = []
        for i, (x1, x2, y) in enumerate(itertools.cycle(data)):
            if step > n_max_iter:
                break

            batch.append((x1, x2, y))
            if len(batch) == batch_size:
                # take one training step

                # pad input and output
                inputs1 = pad_sequence(batch_size, max_input_t, [x1 for x1, x2, y in batch], n_symbols, encode, '0')
                inputs2 = pad_sequence(batch_size, max_input_t, [x2 for x1, x2, y in batch], n_symbols, encode, '0')
                outputs = pad_sequence(batch_size, max_output_t, [y for x1, x2, y in batch], n_symbols, encode, '0')

                feed = {
                    exp1_ph: inputs1,
                    exp2_ph: inputs2,
                    res_ph: outputs,
                }
                # train_op.run(feed_dict=feed)

                # evaluate
                if step > 0 and step % n_eval_interval == 0:
                    pred_vals = prediction.eval(feed_dict=feed)

                    abs_loss = 0.
                    carries = encoder_final_state.eval(feed_dict=feed)
                    ds = encoder_outputs.eval(feed_dict=feed)[:, 0]
                    # convert to number
                    for i, pred_val in enumerate(pred_vals):
                        gt = int(batch[i][-1][::-1], 2)
                        pred_n = int(''.join(convert_output(alphabet, pred_val))[::-1], 2)
                        al = np.abs(gt - pred_n)
                        abs_loss += al
                        print batch[i], bin(gt), bin(pred_n), '|', gt, pred_n
                        print ds[i], carries[i]
                        print

                    print 'abs loss', abs_loss / batch_size
                    print 'perplexity', np.exp(loss.eval(feed_dict=feed))
                    writer.add_summary(summary_op.eval(feed_dict={
                        exp1_ph: inputs1,
                        exp2_ph: inputs2,
                        res_ph: outputs,
                        abs_loss_ph: abs_loss / batch_size,
                    }), global_step.eval())
                    saver.save(sess, checkpoint_dir + '/model', global_step=global_step.eval())

                batch = []
                step += 1
            # print encoder_outputs.eval(feed_dict=feed)

if __name__ == '__main__':
    main()
