import tensorflow as tf
from Tensorflow_DataConverter.load.converter_lab import labels_to_binary


class Model:
    def load_data(self, train_data_set, test_data_set):
        train_data_set._labels = labels_to_binary(train_data_set.labels)
        test_data_set._labels = labels_to_binary(test_data_set.labels)
        self.train_set = train_data_set
        self.test_set = test_data_set

    def train(self):

        # Placeholder
        x = tf.placeholder(tf.float32, [None, 48,28], name='x')
        y_ = tf.placeholder(tf.float32, [None, 11], name='y_')

        #Variable
        W = tf.Variable(tf.zeros([48*28, 11]))
        b = tf.Variable(tf.zeros([11]))

        #Graph
        x_reshape = tf.reshape(x, shape=[-1,48*28], name='x_reshape')
        z = tf.matmul(x_reshape, W) + b
        y = tf.nn.softmax(z)
        cross_entropy = tf.reduce_mean(
              tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

        #Session
        init = tf.global_variables_initializer()

        sess = tf.Session()
        sess.run(init)

        for i in range(5000):
            batch_xs, batch_ys = self.train_set.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        #Evaluation
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: self.test_set.images,
                                            y_: self.test_set.labels}))