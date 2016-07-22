import tensorflow as tf
import numpy as np
import sys, os, time, itertools

def restore_vars(saver, sess, checkpoint_dir):
    """ Restore saved net, global score and step, and epsilons OR
    create checkpoint directory for later storage. """
    sess.run(tf.initialize_all_variables())

    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except OSError:
            pass

    path = tf.train.latest_checkpoint(checkpoint_dir)
    if path is None:
        print 'no existing checkpoint found'
        return False
    else:
        print 'restoring from %s' % path
        saver.restore(sess, path)
        return True

def pad_sequence(batch_size, max_seq_len, seqs, n_symbols, encode):
    # pad input and output
    batch = np.zeros((batch_size, max_seq_len, n_symbols))
    for i in xrange(batch_size):
        seq = seqs[i]
        batch[i, :len(seq)] = [encode(z) for z in seq]
    return batch

def convert_output(alphabet, logits):
    return [alphabet[np.argmax(logit)] for logit in logits]

def main():
    np.random.seed(123)
    tf.set_random_seed(1234)

    # alphabet
    alphabet = '0123456789+-x(),$'
    n_symbols = len(alphabet)
    encode = lambda y: map(lambda x: float(x==y), alphabet)

    # inputs
    batch_size = 128
    # max_input_t = max(input_seq_lens)
    # max_output_t = max(output_seq_lens)
    max_input_t = 15
    max_output_t = 5

    # build model
    exp_ph = tf.placeholder('float', (batch_size, max_input_t, n_symbols))
    res_ph = tf.placeholder('float', (batch_size, max_output_t, n_symbols))
    input_seq_lens_ph = tf.placeholder('int32', (batch_size, ))
    output_seq_lens_ph = tf.placeholder('int32', (batch_size, ))
    abs_loss_ph = tf.placeholder('float')
    valid_rate_ph = tf.placeholder('float', name='valid_rate')
    indicator_ph = tf.placeholder('float', (batch_size, max_output_t), name='indicator')

    encoder = tf.nn.rnn_cell.BasicRNNCell(128)
    decoder = tf.nn.rnn_cell.BasicRNNCell(128)
    w_output = tf.Variable(tf.contrib.layers.xavier_initializer()((128, n_symbols)))

    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder, exp_ph, input_seq_lens_ph, scope='encoder', dtype='float')
    # no input during decoding
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder, tf.zeros((batch_size, max_output_t, n_symbols)), output_seq_lens_ph, initial_state=encoder_final_state, scope='decoder')
    prediction = tf.reshape(tf.matmul(tf.reshape(decoder_outputs, (-1, 128)), w_output), (batch_size, max_output_t, n_symbols))

    # loss with indicators
    loss = tf.reduce_sum(tf.reshape(indicator_ph, (-1,)) * tf.nn.softmax_cross_entropy_with_logits(tf.reshape(prediction, (-1, n_symbols)), tf.reshape(res_ph, (-1, n_symbols)))) / tf.reduce_sum(indicator_ph)

    # optimization
    n_max_iter = int(sys.argv[2])
    n_eval_interval = 1
    n_train_step = 10**3
    global_step = tf.Variable(0, trainable=False, name='global_step')
    initial_learning_rate = 0.001
    decay_steps = 64
    decay_rate = 0.9
    momentum = 0.5
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # summary
    tf.scalar_summary('loss', loss)
    tf.scalar_summary('abs_loss', abs_loss_ph)
    tf.scalar_summary('valid_rate', valid_rate_ph)
    tf.scalar_summary('perplexity', tf.exp(loss))
    tf.scalar_summary('learning_rate', learning_rate)
    # tf.histogram_summary('gradient', optimizer.compute_gradients(loss, var_list=[w12])[0][0])
    summary_op = tf.merge_all_summaries()

    # saver
    checkpoint_dir = sys.argv[3]
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

    # read data
    data = []
    with open(sys.argv[1]) as fi:
        for l in fi:
            x, y = l.strip().split()
            # append <EOS> symbol `$` to the end
            data.append((x, y + '$'))

    with tf.Session() as sess:
        restore_vars(saver, sess, checkpoint_dir)
        writer = tf.train.SummaryWriter('tf-log/%s-%d' % (sys.argv[3], time.time()), sess.graph)

        step = 0
        batch = []
        for i, (x, y) in enumerate(itertools.cycle(data)):
            if step > n_max_iter:
                break

            batch.append((x, y))
            if len(batch) == batch_size:
                # take one training step

                # pad input and output
                input_seq_lens = map(lambda x: len(x[0]), batch)
                output_seq_lens = map(lambda x: len(x[1]), batch)
                inputs = pad_sequence(batch_size, max_input_t, [x for x, y in batch], n_symbols, encode)
                outputs = pad_sequence(batch_size, max_output_t, [y for x, y in batch], n_symbols, encode)
                indicators = [[1.] * len(y) + [0.] * (max_output_t - len(y)) for x, y in batch]
                # print indicators.shape
                feed = {
                    exp_ph: inputs,
                    input_seq_lens_ph: input_seq_lens,
                    res_ph: outputs,
                    output_seq_lens_ph: output_seq_lens,
                    indicator_ph: indicators,
                }

                train_op.run(feed_dict=feed)

                # evaluate
                if step > 0 and step % n_eval_interval == 0:
                    pred_vals = prediction.eval(feed_dict=feed)

                    abs_loss = 0.
                    n_valid_output = 0
                    for i, pred_val in enumerate(pred_vals):
                        pred_int = ''.join(convert_output(alphabet, pred_val))
                        try:
                            pred_n = int(pred_int.split('$')[0])
                            al = np.abs(int(batch[i][1][:-1]) - pred_n)
                            abs_loss += al
                            n_valid_output += 1
                        except:
                            pass
                        # print batch[i][1], pred_int, al
                    # print 'abs loss', abs_loss / batch_size
                    # print 'perplexity', np.exp(loss.eval(feed_dict=feed))
                    writer.add_summary(summary_op.eval(feed_dict={
                        exp_ph: inputs,
                        input_seq_lens_ph: input_seq_lens,
                        res_ph: outputs,
                        output_seq_lens_ph: output_seq_lens,
                        indicator_ph: indicators,
                        abs_loss_ph: abs_loss / batch_size,
                        valid_rate_ph: n_valid_output * 1. / batch_size,
                    }), step)
                    saver.save(sess, checkpoint_dir + '/model', global_step=step)

                batch = []
                step += 1
            # print encoder_outputs.eval(feed_dict=feed)


            # print loss.eval(feed_dict={
            #     exp_ph:
            # })
            # print exp1[i], h_prev_val

            # start of output
            # feed = {
            #     exp_ph: [encode('=')],
            #     h_prev: h_prev_val,
            # }
            #
            # res_loss = 0.
            # for i in xrange(len(res1)):
            #     feed = {
            #         exp_ph: [encode('?')],
            #         res_ph: [encode(res1[i])],
            #         h_prev: h_prev_val,
            #     }
            #     o_val = o.eval(feed_dict=feed)
            #     loss_val = loss.eval(feed_dict=feed)
            #     h_prev_val_t = h1.eval(feed_dict=feed)
            #
            #     train_op.run(feed_dict=feed)
            #
            #     # grad = optimizer.compute_gradients(loss, feed_dict=feed)
            #     # optimizer.apply_gradients(grad)
            #
            #     h_prev_val = h_prev_val_t
            #     res_loss += loss_val
            #     print res1[i], alphabet[np.argmax(o_val)], loss_val
            # print step, res_loss / len(res1)
            # print

if __name__ == '__main__':
    main()
