import numpy as np
import tensorflow as tf


def pairwise_distance(point_cloud):
    """Compute pairwise distance of a point cloud. (euclidean distance)

    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)

    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """
    with tf.variable_scope(None, default_name="pairwise_distance"):
        og_batch_size = point_cloud.get_shape().as_list()[0]
        if point_cloud.get_shape().as_list()[2] == 1:
            point_cloud = tf.squeeze(point_cloud, [2])
        # if og_batch_size == 1:
        #     point_cloud = tf.expand_dims(point_cloud, 0)

        # point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1])
        # point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose)
        point_cloud_inner = tf.matmul(point_cloud, point_cloud, transpose_b=True)
        # point_cloud_inner = -2 * point_cloud_inner
        point_cloud_inner = 2 * point_cloud_inner
        # point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
        point_cloud_square = tf.reduce_sum(tf.multiply(point_cloud, point_cloud), axis=-1, keep_dims=True)
        point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
        # point_cloud_square_tranpose = tf.reduce_sum(tf.square(point_cloud_transpose), axis=-2, keep_dims=True)
        # return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
        return point_cloud_inner - point_cloud_square - point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int

    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    with tf.variable_scope(None, default_name="knn"):
        # neg_adj = -adj_matrix
        # _, nn_idx = tf.nn.top_k(neg_adj, k=k)
        _, nn_idx = tf.nn.top_k(adj_matrix, k=k)
        return nn_idx


def get_edge_feature(point_cloud, nn_idx, k=20, concat_feature=True):
    """Construct edge feature for each point
    Args:
      point_cloud: (batch_size, num_points, 1, num_dims)
      nn_idx: (batch_size, num_points, k)
      k: int

    Returns:
      edge features: (batch_size, num_points, k, num_dims)
    """
    with tf.variable_scope(None, default_name="get_edge_feature"):
        og_batch_size = point_cloud.get_shape().as_list()[0]
        if point_cloud.get_shape().as_list()[2] == 1:
            point_cloud = tf.squeeze(point_cloud, [2])
        # if og_batch_size == 1:
        #     point_cloud = tf.expand_dims(point_cloud, 0)

        point_cloud_central = point_cloud

        point_cloud_shape = point_cloud.get_shape()
        batch_size = point_cloud_shape[0].value
        num_points = point_cloud_shape[1].value
        num_dims = point_cloud_shape[2].value

        idx_ = tf.range(batch_size) * num_points  # offset
        idx_ = tf.reshape(idx_, [batch_size, 1, 1])

        point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
        point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx + idx_)
        point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

        point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

        if concat_feature:
            edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
        else:
            edge_feature = point_cloud_neighbors - point_cloud_central
        return edge_feature
