import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net
import my_quantization

slim = tf.contrib.slim
from tensorflow.contrib.model_pruning.python.layers import layers

def placeholder_input(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3), name='input')
    return pointclouds_pl


def placeholder_label(batch_size):
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size), name='label')
    return labels_pl


def get_network(point_cloud, is_training, neighbor=None, bn_decay=None, dynamic=True,
                STN=True, scale=1., concat_fea=True, weight_decay=0.00004):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    bn_decay = bn_decay if bn_decay is not None else 0.9
    with tf.variable_scope("DGCNN"):
        batch_size = point_cloud.get_shape()[0].value
        num_point = point_cloud.get_shape()[1].value
        # point_cloud = tf.squeeze(point_cloud)
        # if batch_size == 1:
        #     point_cloud = tf.expand_dims(point_cloud, 0)
        end_points = {}
        k = 20
        bn_params = {"is_training": is_training,
                     "decay": bn_decay,
                     'epsilon': 1e-3
                     }

        if STN:
            if neighbor is None:
                adj_matrix = tf_util.pairwise_distance(point_cloud)
                nn_idx = tf_util.knn(adj_matrix, k=k)
            else:
                nn_idx = neighbor
            edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

            transform = input_transform_net(edge_feature, is_training, bn_decay, K=3,
                                            weight_decay=weight_decay, scale=scale)

            with tf.variable_scope("Transform"):
                point_cloud_transformed = tf.matmul(point_cloud, transform)

            if dynamic:
                adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
                nn_idx = tf_util.knn(adj_matrix, k=k)
        else:
            point_cloud_transformed = point_cloud
            if neighbor is None:
                adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
                nn_idx = tf_util.knn(adj_matrix, k=k)
            else:
                nn_idx = neighbor

        edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

        with tf.variable_scope("dgcnn1"):
            net = layers.masked_conv2d(edge_feature,
                              # 64,
                              max(int(round(64 * scale)), 32),
                              [1, 1],
                              padding='VALID',
                              stride=1,
                              normalizer_fn=slim.batch_norm,
                              normalizer_params=bn_params,
                              biases_initializer=tf.zeros_initializer(),
                              weights_regularizer=slim.l2_regularizer(weight_decay),
                              activation_fn=tf.nn.relu6)
            net = tf.reduce_max(net, axis=-2, keepdims=True)
            # net = slim.max_pool2d(net, [1, k], stride=1, padding='VALID')
        net1 = net

        if dynamic:
            adj_matrix = tf_util.pairwise_distance(net)
            nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k, concat_feature=concat_fea)

        with tf.variable_scope("dgcnn2"):
            net = layers.masked_conv2d(edge_feature,
                              # 64,
                              max(int(round(64 * scale)), 32),
                              [1, 1],
                              padding='VALID',
                              stride=1,
                              normalizer_fn=slim.batch_norm,
                              normalizer_params=bn_params,
                              biases_initializer=tf.zeros_initializer(),
                              weights_regularizer=slim.l2_regularizer(weight_decay),
                              activation_fn=tf.nn.relu6)
            net = tf.reduce_max(net, axis=-2, keepdims=True)
            # net = slim.max_pool2d(net, [1, k], stride=1, padding='VALID')
        net2 = net

        if dynamic:
            adj_matrix = tf_util.pairwise_distance(net)
            nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k, concat_feature=concat_fea)

        with tf.variable_scope("dgcnn3"):
            net = layers.masked_conv2d(edge_feature,
                              # 64,
                              max(int(round(64 * scale)), 32),
                              [1, 1],
                              padding='VALID',
                              stride=1,
                              normalizer_fn=slim.batch_norm,
                              normalizer_params=bn_params,
                              biases_initializer=tf.zeros_initializer(),
                              weights_regularizer=slim.l2_regularizer(weight_decay),
                              activation_fn=tf.nn.relu6)
            net = tf.reduce_max(net, axis=-2, keepdims=True)
            # net = slim.max_pool2d(net, [1, k], stride=1, padding='VALID')
        net3 = net

        if dynamic:
            adj_matrix = tf_util.pairwise_distance(net)
            nn_idx = tf_util.knn(adj_matrix, k=k)
        edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k, concat_feature=concat_fea)

        with tf.variable_scope("dgcnn4"):
            net = layers.masked_conv2d(edge_feature,
                              # 128,
                              max(int(round(128 * scale)), 32),
                              [1, 1],
                              padding='VALID',
                              stride=1,
                              normalizer_fn=slim.batch_norm,
                              normalizer_params=bn_params,
                              biases_initializer=tf.zeros_initializer(),
                              weights_regularizer=slim.l2_regularizer(weight_decay),
                              activation_fn=tf.nn.relu6)
            net = tf.reduce_max(net, axis=-2, keepdims=True)
            # net = slim.max_pool2d(net, [1, k], stride=1, padding='VALID')
        net4 = net

        with tf.variable_scope("agg"):
            net = layers.masked_conv2d(tf.concat([net1, net2, net3, net4], axis=-1),
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
                              activation_fn=tf.nn.relu6
                              )

            net = tf.reduce_max(net, axis=1, keepdims=True)
            # net = slim.max_pool2d(net, [num_point, 1], stride=1, padding='VALID')

        # MLP on global point cloud vector
        # net = tf.reshape(net, [batch_size, 1, 1, -1])
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
                          scope='fc1',
                          activation_fn=tf.nn.relu6)
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
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
                          scope='fc2',
                          activation_fn=tf.nn.relu6)
        net = slim.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
        net = layers.masked_conv2d(net,
                          40, [1, 1],
                          padding='SAME',
                          stride=1,
                          # normalizer_fn=slim.batch_norm,
                          # normalizer_params=bn_params,
                          biases_initializer=tf.zeros_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          scope='fc3',
                          # activation_fn=tf.nn.relu6,
                          activation_fn=None,
                          )
        net = tf.reshape(net, [batch_size, -1])
        return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
      label: B, """
    labels = tf.one_hot(indices=label, depth=40)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
    classify_loss = tf.reduce_mean(loss)
    return classify_loss


if __name__ == '__main__':
    batch_size = 2
    num_pt = 124
    pos_dim = 3

    input_feed = np.random.rand(batch_size, num_pt, pos_dim)
    label_feed = np.random.rand(batch_size)
    label_feed[label_feed >= 0.5] = 1
    label_feed[label_feed < 0.5] = 0
    label_feed = label_feed.astype(np.int32)

    # # np.save('./debug/input_feed.npy', input_feed)
    # input_feed = np.load('./debug/input_feed.npy')
    # print input_feed

    with tf.Graph().as_default():
        input_pl = placeholder_input(batch_size, num_pt)
        label_pl = placeholder_label(batch_size)
        pos, ftr = get_network(input_pl, True, quant=1)
        tf.contrib.quantize.create_training_graph(
            quant_delay=1)
        my_quantization.create_training_graph(quant_delay=1)

        # loss = get_loss(logits, label_pl, None)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {input_pl: input_feed, label_pl: label_feed}
            pred, end_points = sess.run([pos, ftr], feed_dict=feed_dict)
            print(pred.shape, pred)
