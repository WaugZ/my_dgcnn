import os
import sys
import tensorflow as tf
import importlib
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '..', 'models'))
sys.path.append(os.path.join(BASE_DIR, '..'))
import my_quantization

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--quantize_delay', type=int, default=None, help='Quantization decay, >0 for open [default:0]')
parser.add_argument('--dynamic', type=int, default=-1,
                    help="Whether dynamically compute the distance[<0 for yes else for no]")
parser.add_argument('--stn', type=int, default=-1,
                    help="whether use STN[<0 for yes else for no]")
parser.add_argument('--quantize_bits', type=int, default=None,
                    help="quantization bits, make sure quantize_delay > 0 when use [default None for 8 bits]")
parser.add_argument('--scale', type=float, default=1., help="dgcnn depth scale")
parser.add_argument('--concat', type=int, default=1, help="whether concat neighbor's feature 1 for yes else for no")
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DYNAMIC = True if FLAGS.dynamic < 0 else False
STN = True if FLAGS.stn < 0 else False
QUANTIZE_BITS = FLAGS.quantize_bits
if QUANTIZE_BITS:
    assert FLAGS.quantize_delay and FLAGS.quantize_delay > 0
SCALE = FLAGS.scale
CONCAT = True if FLAGS.concat == 1 else False
# print('dyancmic: ', DYNAMIC)

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(BASE_DIR, '..', 'models', FLAGS.model + '.py')


if __name__ == "__main__":
    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():
        pointclouds_pl = MODEL.placeholder_input(BATCH_SIZE, NUM_POINT)
        labels_pl = MODEL.placeholder_label(BATCH_SIZE)
        if not FLAGS.quantize_delay:
            is_training = tf.placeholder(tf.bool, shape=(), name="is_training")
        else:
            is_training = True

        # Get model
        pred, end_points = MODEL.get_network(pointclouds_pl, is_training,
                                             dynamic=DYNAMIC,
                                             STN=STN,
                                             scale=SCALE,
                                             concat_fea=CONCAT)

        if FLAGS.quantize_delay and FLAGS.quantize_delay > 0:
            quant_scopes = ["DGCNN/get_edge_feature", "DGCNN/get_edge_feature_1", "DGCNN/get_edge_feature_2",
                            "DGCNN/get_edge_feature_3", "DGCNN/get_edge_feature_4", "DGCNN/agg",
                            "DGCNN/transform_net", "DGCNN/Transform", "DGCNN/dgcnn1", "DGCNN/dgcnn2",
                            "DGCNN/dgcnn3", "DGCNN/dgcnn4"]
            if QUANTIZE_BITS and QUANTIZE_BITS > 0:
                tf.contrib.quantize.experimental_create_training_graph(
                    quant_delay=FLAGS.quantize_delay, weight_bits=QUANTIZE_BITS, activation_bits=QUANTIZE_BITS)
                for scope in quant_scopes:
                    my_quantization.experimental_create_training_graph(quant_delay=FLAGS.quantize_delay,
                                                                       scope=scope, weight_bits=QUANTIZE_BITS,
                                                                       activation_bits=QUANTIZE_BITS)
            else:
                tf.contrib.quantize.create_training_graph(
                    quant_delay=FLAGS.quantize_delay)
                for scope in quant_scopes:
                    my_quantization.experimental_create_training_graph(quant_delay=FLAGS.quantize_delay,
                                                                       scope=scope)
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        if flops and params:
            print("FLOPs = {:,}".format(flops.total_float_ops))
            print("#params = {:,}".format(params.total_parameters))
