import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

slim = tf.contrib.slim
from tensorflow.contrib.model_pruning.python.layers import layers


def input_transform_net(edge_feature, is_training, bn_decay=None, K=3,
                        scale=1., weight_decay=0.00004):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
      Return:
        Transformation matrix of size 3xK """
    bn_params = {"is_training": is_training,
                 "decay": bn_decay,
                 }
    with tf.variable_scope(None, default_name="transform_net"):
        batch_size = edge_feature.get_shape()[0].value
        num_point = edge_feature.get_shape()[1].value
        neighbor = edge_feature.get_shape()[2].value

        # input_image = tf.expand_dims(point_cloud, -1)

        net = layers.masked_conv2d(edge_feature,
                          # 64,
                          max(int(round(64 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          activation_fn=tf.nn.relu6,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tconv1')
        net = layers.masked_conv2d(net,
                          # 128,
                          max(int(round(128 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tconv2',
                          activation_fn=tf.nn.relu6)
        net = tf.reduce_max(net, axis=-2, keepdims=True)
        # net = slim.max_pool2d(net, [1, neighbor], stride=1, padding='VALID')
        net = layers.masked_conv2d(net,
                          # 1024,
                          # max(int(round(1024 * scale)), 32),
                          max(int(round(1024 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tconv3',
                          activation_fn=tf.nn.relu6)
        # print(net)
        net = slim.max_pool2d(net, [num_point, 1],
                              padding='VALID', scope='tmaxpool')
        # net = tf.reshape(net, [batch_size, 1, 1, -1])
        # print(net)
        net = layers.masked_conv2d(net,
                          # 512,
                          max(int(round(512 * scale)), 32),
                          [1, 1],
                          padding='SAME',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tfc1',
                          activation_fn=tf.nn.relu6)
        net = layers.masked_conv2d(net,
                          # 256,
                          max(int(round(256 * scale)), 32),
                          [1, 1],
                          padding='SAME',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tfc2',
                          activation_fn=tf.nn.relu6)

        transform = layers.masked_conv2d(net,
                                K * K, [1, 1],
                                padding='SAME',
                                stride=1,
                                normalizer_fn=None,
                                # normalizer_params=bn_params,
                                biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                scope='transform_XYZ',
                                activation_fn=None,
                                # activation_fn=tf.nn.relu6,
                                )
        transform = tf.reshape(transform, [batch_size, K, K])
        return transform


# def input_transform_net(edge_feature, is_training, bn_decay=None, K=3, is_dist=False):
#     """ Input (XYZ) Transform Net, input is BxNx3 gray image
#       Return:
#         Transformation matrix of size 3xK """
#     batch_size = edge_feature.get_shape()[0].value
#     num_point = edge_feature.get_shape()[1].value
#
#     # input_image = tf.expand_dims(point_cloud, -1)
#     net = tf_util.conv2d(edge_feature, 64, [1, 1],
#                          padding='VALID', stride=[1, 1],
#                          bn=True, is_training=is_training,
#                          scope='tconv1', bn_decay=bn_decay, is_dist=is_dist)
#     net = tf_util.conv2d(net, 128, [1, 1],
#                          padding='VALID', stride=[1, 1],
#                          bn=True, is_training=is_training,
#                          scope='tconv2', bn_decay=bn_decay, is_dist=is_dist)
#
#     net = tf.reduce_max(net, axis=-2, keep_dims=True)
#
#     net = tf_util.conv2d(net, 1024, [1, 1],
#                          padding='VALID', stride=[1, 1],
#                          bn=True, is_training=is_training,
#                          scope='tconv3', bn_decay=bn_decay, is_dist=is_dist)
#     net = tf_util.max_pool2d(net, [num_point, 1],
#                              padding='VALID', scope='tmaxpool')
#
#     net = tf.reshape(net, [batch_size, -1])
#     net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
#                                   scope='tfc1', bn_decay=bn_decay, is_dist=is_dist)
#     net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
#                                   scope='tfc2', bn_decay=bn_decay, is_dist=is_dist)
#
#     with tf.variable_scope('transform_XYZ') as sc:
#         # assert(K==3)
#         with tf.device('/cpu:0'):
#             weights = tf.get_variable('weights', [256, K * K],
#                                       initializer=tf.constant_initializer(0.0),
#                                       dtype=tf.float32)
#             biases = tf.get_variable('biases', [K * K],
#                                      initializer=tf.constant_initializer(0.0),
#                                      dtype=tf.float32)
#         biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
#         transform = tf.matmul(net, weights)
#         transform = tf.nn.bias_add(transform, biases)
#
#     transform = tf.reshape(transform, [batch_size, K, K])
#     return transform

def feature_transform_net(inputs, is_training, bn_decay=None, K=64, scale=1., weight_decay=0.00004):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """

    bn_params = {"is_training": is_training,
                 "decay": bn_decay,
                 'epsilon': 1e-3
                 }
    with tf.variable_scope(None, default_name="feature_transform_net"):

        batch_size = inputs.get_shape()[0].value
        num_point = inputs.get_shape()[1].value

        net = slim.conv2d(inputs,
                          # 64,
                          max(int(round(64 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          activation_fn=tf.nn.relu6,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tconv1')

        net = slim.conv2d(net,
                          # 128,
                          max(int(round(128 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          activation_fn=tf.nn.relu6,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tconv2')

        net = slim.conv2d(net,
                          # 128,
                          max(int(round(1024 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          activation_fn=tf.nn.relu6,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tconv3')
        net = slim.max_pool2d(net, [num_point, 1],
                              padding='VALID', scope='tmaxpool')

        # net = tf.reshape(net, [batch_size, -1])

        net = slim.conv2d(net,
                          # 128,
                          max(int(round(512 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          activation_fn=tf.nn.relu6,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tfc1')

        net = slim.conv2d(net,
                          # 128,
                          max(int(round(256 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          activation_fn=tf.nn.relu6,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tfc2')

        transform = slim.conv2d(net,
                                K * K, [1, 1],
                                padding='SAME',
                                stride=1,
                                normalizer_fn=None,
                                # normalizer_params=bn_params,
                                biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                scope='transform_XYZ',
                                activation_fn=None,
                                # activation_fn=tf.nn.relu6,
                                )

        transform = tf.reshape(transform, [batch_size, K, K])
    return transform