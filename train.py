import tensorflow as tf
import numpy as np
import sys, os, time

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

def main():
    np.random.seed(123)
    tf.set_random_seed(1234)

    # alphabet
    alphabet = '0123456789+-x(),'
    encode = lambda y: map(lambda x: float(x==y), alphabet)

    # build model
    n = len(alphabet)
    exp = tf.placeholder('float', (1, n))
    res = tf.placeholder('float', (1, n))
    h_prev = tf.placeholder('float', (1, 128))
    n_grad_steps = 10

    w01 = tf.Variable(tf.contrib.layers.xavier_initializer()((n, 128)))
    w11 = tf.Variable(tf.contrib.layers.xavier_initializer()((128, 128)))
    w12 = tf.Variable(tf.contrib.layers.xavier_initializer()((128, n)))
    outputs = []
    hiddens = []
    for i in xrange(n_grad_steps):
        if i == 0:
            h_prev = tf.constant(0, (1, 128))
        hiddens.append(tf.nn.relu(tf.matmul(exp, w01)) + tf.matmul(h_prev, w11))
        o = tf.matmul(h1, w12)
        h_prev = h1

    # h1_0 = tf.Variable(tf.truncated_normal((128, )))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(o, res, name='xent'))

    # optimization
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
    tf.scalar_summary('learning_rate', learning_rate)
    tf.histogram_summary('gradient', optimizer.compute_gradients(loss, var_list=[w12])[0][0])
    summary_op = tf.merge_all_summaries()

    # saver
    checkpoint_dir = 'test'
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

    with tf.Session() as sess:
        restore_vars(saver, sess, checkpoint_dir)
        writer = tf.train.SummaryWriter('tf-log/%d' % time.time(), sess.graph_def)

        with open('train.txt') as fi:
            for step in xrange(1000):
                exp1, res1 = fi.readline().strip().split()
                # exp1, res1 = 'x(374,x(x(-(x(-79,-470),-691),-975),-(x(+(-(-(-3,614),-(+(852,+(-456,12)),-313)),714),-238),-(208,-546)))) -2037793766708700'.split()

                h_prev_val = [[0.] * 128]
                for i in xrange(len(exp1)):
                    feed = {
                        exp: [encode(exp1[i])],
                        h_prev: h_prev_val,
                    }
                    h_prev_val = h1.eval(feed_dict=feed)
                    # print exp1[i], h_prev_val

                # start of output
                feed = {
                    exp: [encode('=')],
                    h_prev: h_prev_val,
                }
                h_prev_val = h1.eval(feed_dict=feed)

                res_loss = 0.
                for i in xrange(len(res1)):
                    feed = {
                        exp: [encode('?')],
                        res: [encode(res1[i])],
                        h_prev: h_prev_val,
                    }
                    o_val = o.eval(feed_dict=feed)
                    loss_val = loss.eval(feed_dict=feed)
                    h_prev_val_t = h1.eval(feed_dict=feed)

                    train_op.run(feed_dict=feed)

                    # grad = optimizer.compute_gradients(loss, feed_dict=feed)
                    # optimizer.apply_gradients(grad)

                    writer.add_summary(summary_op.eval(feed_dict=feed), step)
                    h_prev_val = h_prev_val_t
                    res_loss += loss_val
                    print res1[i], alphabet[np.argmax(o_val)], loss_val
                print step, res_loss / len(res1)
                print
                saver.save(sess, checkpoint_dir + '/model', global_step=step)

if __name__ == '__main__':
    main()
