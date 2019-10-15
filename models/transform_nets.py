import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

slim = tf.contrib.slim


def input_transform_net(edge_feature, is_training, bn_decay=None, K=3, is_dist=False, weight_decay=0.0):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
      Return:
        Transformation matrix of size 3xK """
    bn_params = {"is_training": is_training,
                 "decay": bn_decay,
                 'epsilon': 1e-3
                 }
    with tf.variable_scope(None, default_name="transform_net"):
        batch_size = edge_feature.get_shape()[0].value
        num_point = edge_feature.get_shape()[1].value
        neighbor = edge_feature.get_shape()[2].value

        # input_image = tf.expand_dims(point_cloud, -1)

        net = slim.conv2d(edge_feature,
                          64, [1, 1],
                          padding='VALID',
                          stride=1,
                          activation_fn=tf.nn.relu6,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tconv1')
        net = slim.conv2d(net,
                          128, [1, 1],
                          padding='VALID',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tconv2',
                          activation_fn=tf.nn.relu6)
        net = tf.reduce_max(net, axis=-2, keep_dims=True)
        # net = slim.max_pool2d(net, [1, neighbor], stride=1, padding='VALID')
        net = slim.conv2d(net,
                          1024, [1, 1],
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
        net = slim.conv2d(net,
                          512, [1, 1],
                          padding='SAME',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tfc1',
                          activation_fn=tf.nn.relu6)
        net = slim.conv2d(net,
                          256, [1, 1],
                          padding='SAME',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='tfc2',
                          activation_fn=tf.nn.relu6)

        transform = slim.conv2d(net,
                                K * K, [1, 1],
                                padding='SAME',
                                stride=1,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=bn_params,
                                biases_initializer=tf.zeros_initializer(),
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                scope='transform_XYZ',
                                activation_fn=None,
                                # activation_fn=tf.nn.relu6,
                                )
        transform = tf.reshape(transform, [batch_size, K, K])
        return transform
