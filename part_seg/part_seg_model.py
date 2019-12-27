import tensorflow as tf
import numpy as np
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../'))
import tf_util
from transform_nets import input_transform_net

slim = tf.contrib.slim


def get_model(point_cloud, input_label, is_training, cat_num, part_num, \
              batch_size, num_point, weight_decay=.00004, bn_decay=None):
    bn_decay = bn_decay if bn_decay is not None else 0.9

    with tf.variable_scope("DGCNN"):
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        input_image = tf.expand_dims(point_cloud, -1)

        k = 20
        bn_params = {"is_training": is_training,
                     "decay": bn_decay,
                     'renorm': True
                     }

        adj = tf_util.pairwise_distance(point_cloud)
        nn_idx = tf_util.knn(adj, k=k)
        edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(point_cloud, transform)

        input_image = tf.expand_dims(point_cloud_transformed, -1)
        adj = tf_util.pairwise_distance(point_cloud_transformed)
        nn_idx = tf_util.knn(adj, k=k)
        edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k)

        # out1 = tf_util.conv2d(edge_feature, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training, weight_decay=weight_decay,
        #                      scope='adj_conv1', bn_decay=bn_decay, is_dist=True)
        out1 = slim.conv2d(edge_feature,
                           64,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=tf.nn.relu6,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='adj_conv1')

        # out2 = tf_util.conv2d(out1, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training, weight_decay=weight_decay,
        #                      scope='adj_conv2', bn_decay=bn_decay, is_dist=True)
        out2 = slim.conv2d(out1,
                           64,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=tf.nn.relu6,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='adj_conv2')

        net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)

        adj = tf_util.pairwise_distance(net_1)
        nn_idx = tf_util.knn(adj, k=k)
        edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

        # out3 = tf_util.conv2d(edge_feature, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training, weight_decay=weight_decay,
        #                      scope='adj_conv3', bn_decay=bn_decay, is_dist=True)
        out3 = slim.conv2d(edge_feature,
                           64,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=tf.nn.relu6,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='adj_conv3')

        # out4 = tf_util.conv2d(out3, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training, weight_decay=weight_decay,
        #                      scope='adj_conv4', bn_decay=bn_decay, is_dist=True)
        out4 = slim.conv2d(out3,
                           64,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=tf.nn.relu6,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='adj_conv4')

        net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)

        adj = tf_util.pairwise_distance(net_2)
        nn_idx = tf_util.knn(adj, k=k)
        edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

        # out5 = tf_util.conv2d(edge_feature, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training, weight_decay=weight_decay,
        #                      scope='adj_conv5', bn_decay=bn_decay, is_dist=True)
        out5 = slim.conv2d(edge_feature,
                           64,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=tf.nn.relu6,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='adj_conv5')

        # out6 = tf_util.conv2d(out5, 64, [1,1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training, weight_decay=weight_decay,
        #                      scope='adj_conv6', bn_decay=bn_decay, is_dist=True)

        net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)

        # out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='adj_conv7', bn_decay=bn_decay, is_dist=True)
        out7 = slim.conv2d(tf.concat([net_1, net_2, net_3], axis=-1),
                           1024,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=tf.nn.relu6,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='adj_conv7')

        # out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')
        out_max = slim.max_pool2d(out7, [num_point, 1], stride=1, padding='VALID', scope='maxpool')

        one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
        # one_hot_label_expand = tf_util.conv2d(one_hot_label_expand, 64, [1, 1],
        #                      padding='VALID', stride=[1,1],
        #                      bn=True, is_training=is_training,
        #                      scope='one_hot_label_expand', bn_decay=bn_decay, is_dist=True)
        one_hot_label_expand = slim.conv2d(one_hot_label_expand,
                                           64,
                                           # max(int(round(64 * scale)), 32),
                                           [1, 1],
                                           padding='VALID',
                                           stride=1,
                                           normalizer_fn=slim.batch_norm,
                                           normalizer_params=bn_params,
                                           biases_initializer=tf.zeros_initializer(),
                                           weights_regularizer=slim.l2_regularizer(weight_decay),
                                           activation_fn=tf.nn.relu6,
                                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                                           scope='one_hot_label_expand')
        out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])
        expand = tf.tile(out_max, [1, num_point, 1, 1])

        concat = tf.concat(axis=3, values=[expand,
                                           net_1,
                                           net_2,
                                           net_3])

        # net2 = tf_util.conv2d(concat, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
        #           bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay, is_dist=True)
        net2 = slim.conv2d(concat,
                           256,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=tf.nn.relu6,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='seg/conv1')
        # net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
        net2 = slim.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp1')
        # net2 = tf_util.conv2d(net2, 256, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
        #           bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay, is_dist=True)
        net2 = slim.conv2d(net2,
                           256,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=tf.nn.relu6,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='seg/conv2')
        # net2 = tf_util.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
        net2 = slim.dropout(net2, keep_prob=0.6, is_training=is_training, scope='seg/dp2')
        # net2 = tf_util.conv2d(net2, 128, [1,1], padding='VALID', stride=[1,1], bn_decay=bn_decay,
        #           bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay, is_dist=True)
        net2 = slim.conv2d(net2,
                           128,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=slim.batch_norm,
                           normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=tf.nn.relu6,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='seg/conv3')
        # net2 = tf_util.conv2d(net2, part_num, [1,1], padding='VALID', stride=[1,1], activation_fn=None,
        #           bn=False, scope='seg/conv4', weight_decay=weight_decay, is_dist=True)
        net2 = slim.conv2d(net2,
                           part_num,
                           # max(int(round(64 * scale)), 32),
                           [1, 1],
                           padding='VALID',
                           stride=1,
                           normalizer_fn=None,
                           # normalizer_params=bn_params,
                           biases_initializer=tf.zeros_initializer(),
                           weights_regularizer=slim.l2_regularizer(weight_decay),
                           activation_fn=None,
                           weights_initializer=tf.contrib.layers.xavier_initializer(),
                           scope='seg/conv4')

        net2 = tf.reshape(net2, [batch_size, num_point, part_num])

    return net2


def get_loss(seg_pred, seg):
    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg),
                                           axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)
    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res
