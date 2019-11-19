import tensorflow as tf
import os
import sys
import importlib
import argparse

import my_quantization
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--quantize_delay', type=int, default=-1,
                    help='Whether the model is quantized >= 0 for yes[default No]')
parser.add_argument('--output_graph', default="log/infer_graph.pb", help='the path to export graph def')
parser.add_argument('--dynamic', type=int, default=-1,
                    help="Whether dynamically compute the distance[<0 for yes else for no]")
parser.add_argument('--stn', type=int, default=-1,
                    help="whether use STN[<0 for yes else for no]")
parser.add_argument('--neighbor', type=int, default=None,
                    help="whether neighbor is an input of the network[default None for no, else for yes]")
parser.add_argument('--scale', type=float, default=1., help="dgcnn depth scale")
parser.add_argument('--concat', type=int, default=1, help="whether concat neighbor's feature 1 for yes else for no")
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DYNAMIC = True if FLAGS.dynamic < 0 else False
STN = True if FLAGS.stn < 0 else False
MODEL = importlib.import_module(FLAGS.model)
NEIGHBOR = FLAGS.neighbor
if NEIGHBOR:
    assert DYNAMIC is False, "when split the structure of net, must not dynamically find neighbor"
SCALE = FLAGS.scale
CONCAT = True if FLAGS.concat == 1 else False


if __name__ == "__main__":
    with tf.Graph().as_default() as graph:
        pointclouds_pl = MODEL.placeholder_input(BATCH_SIZE, NUM_POINT)
        k = 20
        if NEIGHBOR:
            knn_pl = tf.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_POINT, k), name='knn')
        else:
            knn_pl = None
        MODEL.get_network(pointclouds_pl, neighbor=knn_pl, is_training=False, dynamic=DYNAMIC, STN=STN,
                          scale=SCALE, concat_fea=CONCAT)

        if FLAGS.quantize_delay >= 0:
            quant_scopes = ["DGCNN/get_edge_feature", "DGCNN/get_edge_feature_1", "DGCNN/get_edge_feature_2",
                            "DGCNN/get_edge_feature_3", "DGCNN/get_edge_feature_4", "DGCNN/agg",
                            "DGCNN/transform_net", "DGCNN/Transform", "DGCNN/dgcnn1", "DGCNN/dgcnn2",
                            "DGCNN/dgcnn3", "DGCNN/dgcnn4"]
            tf.contrib.quantize.create_eval_graph()
            for scope in quant_scopes:
                my_quantization.experimental_create_eval_graph(scope=scope)

        graph_def = graph.as_graph_def()

        with tf.gfile.GFile(FLAGS.output_graph, 'wb') as f:
            f.write(graph_def.SerializeToString())
        print("graph def written in {}".format(FLAGS.output_graph))
