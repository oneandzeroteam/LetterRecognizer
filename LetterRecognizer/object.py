import tensorflow as tf


class Weight:
    def __init__(self,
                 W_conv1,
                 W_conv2,
                 W_fc1,
                 W_fc2):
        self.W_conv1 = W_conv1
        self.W_conv2 = W_conv2
        self.W_fc1 = W_fc1
        self.W_fc2 = W_fc2

class Placebundle:
    def __init__(self,
                 x,
                 y_,
                 W,
                 B,
                 keep_prob):
        self.x = x
        self.y_ = y_
        self.W = W
        self.B = B
        self.keep_prob = keep_prob



def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
