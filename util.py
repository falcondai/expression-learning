import tensorflow as tf
import numpy as np
import os

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

def pad_sequence(batch_size, max_seq_len, seqs, n_symbols, encode, pad_symbol=' '):
    # pad input and output
    batch = np.zeros((batch_size, max_seq_len, n_symbols))
    for i in xrange(batch_size):
        seq = seqs[i] + pad_symbol * (max_seq_len - len(seqs[i]))
        batch[i] = [encode(z) for z in seq]
    return batch

def convert_output(alphabet, logits):
    return [alphabet[np.argmax(logit)] for logit in logits]
