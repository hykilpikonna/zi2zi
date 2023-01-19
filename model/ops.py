# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import pdb

import tensorflow as tf


def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm"):
    return tf.keras.layers.BatchNormalization(momentum=decay, epsilon=epsilon,
                                              scale=True)(x, training=is_training)


def conv2d(x, output_filters, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="conv2d"):
    with tf.compat.v1.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.compat.v1.get_variable('W', [kh, kw, shape[-1], output_filters],
                            initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
        Wconv = tf.nn.conv2d(input=x, filters=W, strides=[1, sh, sw, 1], padding='SAME')

        biases = tf.compat.v1.get_variable('b', [output_filters], initializer=tf.compat.v1.constant_initializer(0.0))
        Wconv_plus_b = tf.nn.bias_add(Wconv, biases)
        # Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b


def deconv2d(x, output_shape, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="deconv2d"):
    with tf.compat.v1.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        input_shape = x.get_shape().as_list()
        W = tf.compat.v1.get_variable('W', [kh, kw, output_shape[-1], input_shape[-1]],
                            initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, sh, sw, 1])

        biases = tf.compat.v1.get_variable('b', [output_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
        deconv_plus_b = tf.nn.bias_add(deconv, biases)  # tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv_plus_b


def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def fc(x, output_size, stddev=0.02, scope="fc"):
    with tf.compat.v1.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.compat.v1.get_variable("W", [shape[1], output_size], tf.float32,
                            tf.compat.v1.random_normal_initializer(stddev=stddev))
        b = tf.compat.v1.get_variable("b", [output_size],
                            initializer=tf.compat.v1.constant_initializer(0.0))
        return tf.matmul(x, W) + b


def init_embedding(size, dimension, stddev=0.01, scope="embedding"):
    with tf.compat.v1.variable_scope(scope):
        return tf.compat.v1.get_variable("E", [size, 1, 1, dimension], tf.float32,
                               tf.compat.v1.random_normal_initializer(stddev=stddev))


def conditional_instance_norm(x, ids, labels_num, mixed=False, scope="conditional_instance_norm"):
    with tf.compat.v1.variable_scope(scope):
        shape = x.get_shape().as_list()
        output_filters = shape[-1]
        scale = tf.compat.v1.get_variable("scale", [labels_num, output_filters], tf.float32, tf.compat.v1.constant_initializer(1.0))
        shift = tf.compat.v1.get_variable("shift", [labels_num, output_filters], tf.float32, tf.compat.v1.constant_initializer(0.0))

        mu, sigma = tf.nn.moments(x=x, axes=[1, 2], keepdims=True)
        norm = (x - mu) / tf.sqrt(sigma + 1e-5)
        batch_scale = tf.reshape(tf.nn.embedding_lookup(params=[scale], ids=ids), [-1, 1, 1, output_filters])
        batch_shift = tf.reshape(tf.nn.embedding_lookup(params=[shift], ids=ids), [-1, 1, 1, output_filters])

        z = norm * batch_scale + batch_shift
        return z
