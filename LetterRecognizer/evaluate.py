from config.config import HYPARMS


def fill_feed_dict(data_set, images_pl, labels_pl, keep_prob, evalflag=False):
    images_feed, labels_feed = data_set.next_batch(HYPARMS.batch_size)
    if evalflag:
        dropout_rate = 1
    else:
        dropout_rate = HYPARMS.dropout_rate
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
        keep_prob: dropout_rate,
    }
    return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            keep_prob,
            data_set):

  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // HYPARMS.batch_size
  num_examples = steps_per_epoch * HYPARMS.batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder,
                               keep_prob,
                               True)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
