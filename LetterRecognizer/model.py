# -*- coding: utf-8 -*-

import tensorflow as tf
from config.config import HYPARMS
import os, time
from .evaluate import fill_feed_dict, do_eval
from .graph import placeholder_inputs,graph_model,calcul_loss,training,evaluation


class Model:
    def __init__(self):
        self.batch_size = HYPARMS.batch_size
        self.init()

    def init(self):
        self.sess = tf.Session()
        self.placebundle = placeholder_inputs(self.batch_size)
        self.load_graph()
        #self.load_summary()

        initiate = tf.global_variables_initializer()
        self.sess.run(initiate)
        self.init_saver()

    def load_data(self, train_data_set, test_data_set):
        self.train_set = train_data_set
        self.test_set = test_data_set

    def load_graph(self):
        self.placebundle = placeholder_inputs(HYPARMS.batch_size)
        self.logits = graph_model(self.placebundle)
        self.loss = calcul_loss(self.logits, self.placebundle)
        self.train_op = training(self.loss, HYPARMS.learning_rate)
        self.eval_correct = evaluation(self.logits, self.placebundle)

    def load_summary(self):
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(HYPARMS.log_dir, self.sess.graph)

    def update_summary(self, feed_dict):
        summary_str = self.sess.run(self.summary, feed_dict=feed_dict)
        self.summary_writer.add_summary(summary_str, self.step)
        self.summary_writer.flush()

    def init_saver(self):
        self.checkpoint_file = os.path.join(HYPARMS.ckpt_dir, HYPARMS.ckpt_name)
        saver_map = {
            "w_conv1": self.placebundle.W.W_conv1,
            "w_conv2": self.placebundle.W.W_conv2,
            "w_fc1": self.placebundle.W.W_fc1,
            "w_fc2": self.placebundle.W.W_fc2,
            "b_conv1": self.placebundle.B.W_conv1,
            "b_conv2": self.placebundle.B.W_conv2,
            "b_fc1": self.placebundle.B.W_fc1,
            "b_fc2": self.placebundle.B.W_fc2,
                     }
        self.saver = tf.train.Saver(saver_map)
        if not tf.gfile.Exists(HYPARMS.ckpt_dir):
            tf.gfile.MakeDirs(HYPARMS.ckpt_dir)

    def save_saver(self):
        self.saver.save(self.sess, self.checkpoint_file, global_step=self.step)

    def restore_saver(self):
        ckpt = tf.train.get_checkpoint_state(HYPARMS.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model loaded")
        else:
            print("No checkpoint file found")

    def train(self):
        start_time = time.time()
        for step in range(HYPARMS.max_steps):
            step_start_time = time.time()

            feed_dict = fill_feed_dict(self.train_set,
                                       self.placebundle.x,
                                       self.placebundle.y_,
                                       self.placebundle.keep_prob)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = self.sess.run([self.train_op, self.loss],
                                     feed_dict=feed_dict)

            step_duration = time.time() - step_start_time
            if step % int(HYPARMS.max_steps/50) == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, step_duration))
                # Update the events file.
                self.step = step
                #self.update_summary(feed_dict)

            # Save a checkpoint
            # and evaluate the model periodically.

            if (step + 1) % int(HYPARMS.max_steps/10) == 0 or (step + 1) == HYPARMS.max_steps:
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(self.sess,
                        self.eval_correct,
                        self.placebundle.x,
                        self.placebundle.y_,
                        self.placebundle.keep_prob,
                        self.train_set)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(self.sess,
                        self.eval_correct,
                        self.placebundle.x,
                        self.placebundle.y_,
                        self.placebundle.keep_prob,
                        self.test_set)

        duration = time.time() - start_time
        print('Duration = (%.3f sec)' % (duration))
        self.step = step
        self.save_saver()


class RecogModel(Model):
    def __init__(self):
        self.batch_size = 1
        self.init()
        self.restore_saver()

    def load_graph(self):
        self.placebundle = placeholder_inputs(self.batch_size)
        self.logits = graph_model(self.placebundle)
        self.sftmax = tf.nn.softmax(self.logits)
        self.classified = tf.argmax(self.sftmax, 1)

    def init(self):
        super().init()

    def restore_saver(self):
        super().restore_saver()

    def init_saver(self):
        super().init_saver()

    def save_saver(self):
        super().save_saver()

    def predict(self, input):
        input = input.reshape([1,input.shape[0],input.shape[1]])
        with self.sess.as_default():
            return self.classified.eval(feed_dict = {self.placebundle.x: input,
                                                    self.placebundle.keep_prob: 1})
