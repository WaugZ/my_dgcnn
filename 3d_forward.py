import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2
import numpy as np
import os
import sys
import random
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--graph_def', default=None, help='The pb graph file')
parser.add_argument('--input', default='input', help='The name of input node')
parser.add_argument('--output', default='DGCNN/Reshape', help='The name of output node')
parser.add_argument('--shape', type=str, default='1,1024,3', help='The shape of input')
parser.add_argument('--mean_val', type=int, default=0, help='Mean value of input[default:0]')
parser.add_argument('--std_val', type=int, default=1, help='Standard dev value of input[default:1]')
parser.add_argument('--eval_num', type=int, default=1, help='How many samples want to forward[default:1]')
parser.add_argument('--output_graph', default=None, help='The path to write output graph def')
FLAGS = parser.parse_args()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
# NUM_POINT = 1024

def get_output(pb_file, input_node, output_node, input_shape, input_quant=(0., 1.)):
    """
    forward infer the graph, record the ever max or min value of the nodes need quantization
    """
    NUM_POINT = input_shape[1]
    with tf.Graph().as_default() as graph:  # Set default graph as graph
        with tf.Session() as sess:
            # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
            # Shuffle train files
            test_file_idxs = np.arange(0, len(TEST_FILES))
            np.random.shuffle(test_file_idxs)

            with gfile.FastGFile(pb_file, 'rb') as f:
                current_data, current_label = provider.loadDataFile(TEST_FILES[test_file_idxs[0]])
                current_data = current_data[:, 0:NUM_POINT, :]
                current_label = np.squeeze(current_label)
                # print(current_data.shape)

                file_size = current_data.shape[0]

                indx = random.randint(0, file_size)
                data = current_data[indx, :, :]
                label = current_label[indx]

                # quant
                data = (data - input_quant[0]) / input_quant[1]

                # Set FCN graph to the default graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()

                # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="",
                    op_dict=None,
                    producer_op_list=None
                )

                nodes = {}

                # initialize_all_variables
                tf.global_variables_initializer()
                # init_input = None
                input_ = graph.get_tensor_by_name(input_node)
                output_ = graph.get_tensor_by_name(output_node)
                Session_out = sess.run(output_, feed_dict={input_.name: [data]})
                print(Session_out)
                print("predict: {}".format(np.argmax(Session_out)), " ground truth: {}".format(label))


def forward(pb_file, input_node, input_shape, input_quant=(0., 1.), mins={}, maxes={}):
    """
    forward infer the graph, record the ever max or min value of the nodes need quantization
    """
    NUM_POINT = input_shape[1]
    with tf.Graph().as_default() as graph:  # Set default graph as graph
        with tf.Session() as sess:
            # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
            # Shuffle train files
            test_file_idxs = np.arange(0, len(TEST_FILES))
            np.random.shuffle(test_file_idxs)

            with gfile.FastGFile(pb_file, 'rb') as f:
                current_data, current_label = provider.loadDataFile(TEST_FILES[test_file_idxs[0]])
                current_data = current_data[:, 0:NUM_POINT, :]
                current_label = np.squeeze(current_label)
                # print(current_data.shape)

                file_size = current_data.shape[0]

                indx = random.randint(0, file_size)
                data = current_data[indx, :, :]
                label = current_label[indx]

                # quant
                data = (data - input_quant[0]) / input_quant[1]

                # Set FCN graph to the default graph
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                sess.graph.as_default()

                # Import a graph_def into the current default Graph (In this case, the weights are (typically) embedded in the graph)
                tf.import_graph_def(
                    graph_def,
                    input_map=None,
                    return_elements=None,
                    name="",
                    op_dict=None,
                    producer_op_list=None
                )

                nodes = {}

                # initialize_all_variables
                tf.global_variables_initializer()
                # init_input = None
                for ind, op in enumerate(graph.get_operations()):
                    # INFERENCE Here
                    # print(op.inputs, op.outputs)
                    feed_dict = {}
                    if ind == 0:
                        # init_input = graph.get_tensor_by_name(input_node)
                        nodes[input_node] = [data]
                    for inp in op.inputs:
                        # print(inp)
                        t = graph.get_tensor_by_name(inp.name)
                        # print(t.op.type)
                        if t.op.type != "Const":
                            feed_dict[t] = nodes[t.name]
                    # print(op.outputs)
                    if not op.outputs: continue
                    l_output = graph.get_tensor_by_name(op.outputs[0].name)  # Output Tensor

                    if len(feed_dict) > 0:
                        # if ind == len(ops) - 1:
                        #   feed_dict.setdefault(init_input, [image])
                        Session_out = sess.run(l_output, feed_dict=feed_dict)
                        # Session_out1 = sess.run(l_output, feed_dict={init_input: [image]})
                        if op.outputs[0] not in nodes:
                            nodes[op.outputs[0].name] = Session_out

                for name, tensor in nodes.items():
                    max_v = np.max(tensor)
                    min_v = np.min(tensor)
                    if name in maxes:
                        if maxes[name] < max_v:
                            maxes[name] = max_v
                    else:
                        maxes[name] = max_v
                    if name in mins:
                        if mins[name] < min_v:
                            mins[name] = min_v
                    else:
                        mins[name] = min_v
            return mins, maxes


if __name__ == "__main__":
    pbfile = FLAGS.graph_def
    input_node = FLAGS.input
    input_node = input_node + ":0"
    output_node = FLAGS.output
    output_node = output_node + ":0"
    shape_tmp = FLAGS.shape
    shape = [int(s) for s in shape_tmp.split(',')]
    mean = FLAGS.mean_val
    dev = FLAGS.std_val
    loop = FLAGS.eval_num
    output_graph = FLAGS.output_graph
    if not output_graph:
        for i in range(loop):
            get_output(pbfile, input_node, output_node, shape, (mean, dev))
    else:
        if not os.path.isdir(os.path.dirname(output_graph)):
            print("Invalid path: {}".format(output_graph))
            exit(-1)

        input_graph_def = graph_pb2.GraphDef()

        mins = {}
        maxes = {}
        for i in range(loop):
            forward(pbfile, input_node, shape, (mean, dev), mins, maxes)
        with gfile.Open(pbfile, "rb") as f:
            data = f.read()
            input_graph_def.ParseFromString(data)

        output_graph_def = get_quant_graph_def(input_graph_def, mins, maxes)
        f = gfile.FastGFile(output_graph, 'w')
        f.write(output_graph_def.SerializeToString())
        print("transform finish.")

