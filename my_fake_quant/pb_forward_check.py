import os
import sys
import tensorflow as tf  # Default graph is initialized when the library is imported
import numpy as np
from tensorflow.python.platform import gfile
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import provider
import pc_util

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
NUM_POINT = 1024
NUM_CLASSES = 40
DUMP_DIR = '../dump'
BATCH_SIZE = 1
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]


def show_graph(pb_file):
    with tf.Graph().as_default() as graph:  # Set default graph as graph
        with tf.Session() as sess:
            # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
            with gfile.FastGFile(pb_file, 'rb') as f:
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

                # Print the name of operations in the session
                for op in graph.get_operations():
                    print("Operation Name :", op.name)  # Operation name
                    print("Tensor Stats :", str(op.values()))  # Tensor name


def infer_graph(pb_file, input_node, output_nodes):
    if not isinstance(output_nodes, list):
        output_nodes = [output_nodes]
    with tf.Graph().as_default() as graph:  # Set default graph as graph
        with tf.Session() as sess:
            # Load the graph in graph_def
            # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
            with gfile.FastGFile(pb_file, 'rb') as f:
                for fn in range(len(TEST_FILES)):
                    print('----' + str(fn) + '----')
                    current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
                    current_data = current_data[:, 0:NUM_POINT, :]
                    # current_data = np.expand_dims(current_data, axis=-2)
                    current_label = np.squeeze(current_label)
                    print(current_data.shape)

                    file_size = current_data.shape[0]
                    num_batches = file_size // BATCH_SIZE
                    print(file_size)

                    num_votes = 1
                    total_correct = 0
                    total_seen = 0
                    for f_idx in range(file_size):
                        for vote_idx in range(num_votes):
                            # rotated_data = provider.rotate_point_cloud_by_angle(current_data[f_idx:f_idx+1, :, :],
                            #                                                     vote_idx / float(num_votes) * np.pi * 2)

                            data = current_data[f_idx:f_idx + 1, :, :]
                            label = current_label[f_idx]
                            # interpreter.set_tensor(input_details[0]['index'], current_data[f_idx:f_idx+1, :, :])
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

                            res = []
                            for output_node in output_nodes:
                                # INFERENCE Here
                                l_input = graph.get_tensor_by_name(input_node)  # Input Tensor
                                l_output = graph.get_tensor_by_name(output_node)  # Output Tensor

                                # initialize_all_variables
                                tf.global_variables_initializer()

                                Session_out = sess.run(l_output, feed_dict={l_input: data})
                                res.append(Session_out)
                                # return res
                                print(Session_out, np.argmax(Session_out), label)
                                pred = np.argmax(Session_out)

                                # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
                                total_correct += 1 if pred == label else 0
                                total_seen += 1

                    print('eval accuracy: %f' % (total_correct / float(total_seen)))
                    # print('eval avg class acc: %f' % (
                    #     np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))


show_graph("/media/wangzi/wangzi/codes/my_dgcnn/log_0828_best_quant_ori/dgcnn_quant.pb")
infer_graph("/media/wangzi/wangzi/codes/my_dgcnn/log_0828_best_quant_ori/dgcnn_quant.pb", 'input:0', 'DGCNN/Reshape:0')
