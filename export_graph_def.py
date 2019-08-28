import tensorflow as tf
import os
import sys
import importlib
import argparse

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
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)

if __name__ == "__main__":
    with tf.Graph().as_default() as graph:
        pointclouds_pl = MODEL.placeholder_input(BATCH_SIZE, NUM_POINT)

        MODEL.get_network(pointclouds_pl, is_training=False)

        if FLAGS.quantize_delay >= 0:
            tf.contrib.quantize.create_eval_graph()

        graph_def = graph.as_graph_def()

        with tf.gfile.GFile(FLAGS.output_graph, 'wb') as f:
            f.write(graph_def.SerializeToString())
        print("graph def written in {}".format(FLAGS.output_graph))
