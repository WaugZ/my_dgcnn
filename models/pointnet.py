import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net

slim = tf.contrib.slim

def placeholder_input(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3), name='input')
    return pointclouds_pl


def placeholder_label(batch_size):
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size), name='label')
    return labels_pl


def get_network(point_cloud, is_training, bn_decay=None, neighbor=None, dynamic=True,
                STN=True, scale=1., concat_fea=True, weight_decay=0.00004):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    bn_decay = bn_decay if bn_decay is not None else 0.9
    bn_params = {"is_training": is_training,
                 "decay": bn_decay,
                 }

    with tf.variable_scope('PointNet'):

        edge_feature = tf.expand_dims(point_cloud, -2)
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(edge_feature, is_training, bn_decay, K=3,
                                            weight_decay=weight_decay, scale=scale)
        point_cloud_transformed = tf.matmul(point_cloud, transform)
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = slim.conv2d(input_image,
                          # 64,
                          max(int(round(64 * scale)), 32),
                          [1, 3],
                          padding='VALID',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          activation_fn=tf.nn.relu6,
                          scope='conv1')

        net = slim.conv2d(net,
                          # 64,
                          max(int(round(64 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          activation_fn=tf.nn.relu6,
                          scope='conv2')

        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])

        net = slim.conv2d(net_transformed,
                          # 64,
                          max(int(round(64 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          activation_fn=tf.nn.relu6,
                          scope='conv3')

        net = slim.conv2d(net,
                          # 64,
                          max(int(round(128 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          activation_fn=tf.nn.relu6,
                          scope='conv4')

        net = slim.conv2d(net,
                          # 64,
                          max(int(round(1024 * scale)), 32),
                          [1, 1],
                          padding='VALID',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          activation_fn=tf.nn.relu6,
                          scope='conv5')

        # Symmetric function: max pooling
        net = slim.max_pool2d(net, [num_point, 1],
                              padding='VALID', scope='maxpool')

        # net = tf.reshape(net, [batch_size, -1])

        net = slim.conv2d(net,
                          # 512,
                          max(int(round(512 * scale)), 32),
                          [1, 1],
                          padding='SAME',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='fc1',
                          activation_fn=tf.nn.relu6)
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
        net = slim.conv2d(net,
                          # 512,
                          max(int(round(256 * scale)), 32),
                          [1, 1],
                          padding='SAME',
                          stride=1,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='fc2',
                          activation_fn=tf.nn.relu6)
        net = slim.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
        net = slim.conv2d(net,
                          40,
                          [1, 1],
                          padding='SAME',
                          stride=1,
                          # normalizer_fn=slim.batch_norm,
                          normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='fc3',
                          activation_fn=None)
        net = tf.reshape(net, [batch_size, -1])
    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)