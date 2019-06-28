# coding=utf-8
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
from six.moves import range
import numpy as np

def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_size + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (le - 1) / 2) * (j - (ls - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


def add_gradient_noise(t, stddev=1e-3, name=None):
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)


def zero_nil_slot(t, name=None):
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.pack([1, s]))
        return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0, name=name)


class MemN2N_KV(object):
    def __init__(self, batch_size, vocab_size,
                 query_size, story_size, memory_key_size,
                 memory_value_size, embedding_size,
                 feature_size=30,
                 hops=3,
                 reader='bow',
                 l2_lambda=0.2,
                 name='KeyValueMemN2N'):
        self._story_size = story_size
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._query_size = query_size
        self._memory_key_size = memory_key_size
        self._embedding_size = embedding_size
        self._hops = hops
        self._name = name
        self._memory_value_size = memory_value_size
        self._encoding = tf.constant(position_encoding(self._story_size, self._embedding_size), name="encoding")
        self._reader = reader
        self._build_inputs()

        d = feature_size
        self._feature_size = feature_size
        self._n_hidden = feature_size
        self.reader_feature_size = 0
        # trainable variables
        if reader == 'bow':
            self.reader_feature_size = self._embedding_size


        self.A = tf.get_variable('A', shape=[self._feature_size, self.reader_feature_size],
                                 initializer=tf.contrib.layers.xavier_initializer())
        self.TK = tf.get_variable('TK', shape=[self._memory_value_size, self.reader_feature_size],
                                  initializer=tf.contrib.layers.xavier_initializer())
        self.TV = tf.get_variable('TV', shape=[self._memory_value_size, self.reader_feature_size],
                                  initializer=tf.contrib.layers.xavier_initializer())

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            nil_word_slot = tf.zeros([1, embedding_size])
            self.W = tf.concat([nil_word_slot,
                                tf.get_variable('W', shape=[vocab_size - 1, embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())], 0)
            self.W_memory = tf.concat(
                [nil_word_slot, tf.get_variable('W_memory', shape=[vocab_size - 1, embedding_size],
                                                initializer=tf.contrib.layers.xavier_initializer())], 0)
            # self.W_memory = self.W
            self._nil_vars = set([self.W.name, self.W_memory.name])
            # shape: [batch_size, query_size, embedding_size]
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self._query)
            # shape: [batch_size, memory_size, story_size, embedding_size]
            self.mkeys_embedded_chars = tf.nn.embedding_lookup(self.W_memory, self._memory_key)
            # shape: [batch_size, memory_size, story_size, embedding_size]
            self.mvalues_embedded_chars = tf.nn.embedding_lookup(self.W_memory, self._memory_value)

        if reader == 'bow':
            q_r = tf.reduce_sum(self.embedded_chars * self._encoding, 1)
            doc_r = tf.reduce_sum(self.mkeys_embedded_chars * self._encoding, 2)
            value_r = tf.reduce_sum(self.mvalues_embedded_chars * self._encoding, 2)

        r_list = []
        for _ in range(self._hops):
            # define R for variables
            R = tf.get_variable('R{}'.format(_), shape=[self._feature_size, self._feature_size],
                                initializer=tf.contrib.layers.xavier_initializer())
            r_list.append(R)

        o = self._key_addressing(doc_r, value_r, q_r, r_list)
        o = tf.transpose(o)
        if reader == 'bow':
            # self.B = self.A
            self.B = tf.get_variable('B', shape=[self._feature_size, self.reader_feature_size],
                                     initializer=tf.contrib.layers.xavier_initializer())


        y_tmp = tf.matmul(self.B, self.W_memory, transpose_b=True)
        with tf.name_scope("prediction"):
            logits = tf.matmul(o, y_tmp)  # + logits_bias

            probs = tf.nn.softmax(tf.cast(logits, tf.float32))

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                                       labels=tf.cast(self._labels, tf.float32),
                                                                       name='cross_entropy')
            cross_entropy_sum = tf.reduce_sum(cross_entropy, name="cross_entropy_sum")

            # loss op
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])
            loss_op = cross_entropy_sum + l2_lambda * lossL2
            # predict ops
            predict_op = tf.argmax(probs, 1, name="predict_op")

            # assign ops
            self.loss_op = loss_op
            self.predict_op = predict_op
            self.probs = probs

    def _build_inputs(self):
        with tf.name_scope("input"):
            self._memory_key = tf.placeholder(tf.int32, [None, self._memory_value_size, self._story_size],
                                              name='memory_key')

            self._query = tf.placeholder(tf.int32, [None, self._query_size], name='question')

            self._memory_value = tf.placeholder(tf.int32, [None, self._memory_value_size, self._story_size],
                                                name='memory_value')

            self._labels = tf.placeholder(tf.float32, [None, self._vocab_size], name='answer')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    '''
    mkeys: the vector representation for keys in memory
    -- shape of each mkeys: [1, embedding_size]
    mvalues: the vector representation for values in memory
    -- shape of each mvalues: [1, embedding_size]
    questions: the vector representation for the question
    -- shape of questions: [1, embedding_size]
    -- shape of R: [feature_size, feature_size]
    -- shape of self.A: [feature_size, embedding_size]
    -- shape of self.B: [feature_size, embedding_size]
    self.A, self.B and R are the parameters to learn
    '''

    def _key_addressing(self, mkeys, mvalues, questions, r_list):

        with tf.variable_scope(self._name):
            u = tf.matmul(self.A, questions, transpose_b=True)
            u = [u]
            for _ in range(self._hops):
                R = r_list[_]
                u_temp = u[-1]
                mk_temp = mkeys + self.TK
                k_temp = tf.reshape(tf.transpose(mk_temp, [2, 0, 1]), [self.reader_feature_size, -1])
                a_k_temp = tf.matmul(self.A, k_temp)
                a_k = tf.reshape(tf.transpose(a_k_temp), [-1, self._memory_key_size, self._feature_size])
                u_expanded = tf.expand_dims(tf.transpose(u_temp), [1])
                dotted = tf.reduce_sum(a_k * u_expanded, 2)
                probs = tf.nn.softmax(dotted)
                probs_expand = tf.expand_dims(probs, -1)
                mv_temp = mvalues + self.TV
                v_temp = tf.reshape(tf.transpose(mv_temp, [2, 0, 1]), [self.reader_feature_size, -1])
                a_v_temp = tf.matmul(self.A, v_temp)
                a_v = tf.reshape(tf.transpose(a_v_temp), [-1, self._memory_key_size, self._feature_size])
                o_k = tf.reduce_sum(probs_expand * a_v, 1)
                o_k = tf.transpose(o_k)
                u_k = tf.matmul(R, u[-1] + o_k)
                u.append(u_k)

            return u[-1]