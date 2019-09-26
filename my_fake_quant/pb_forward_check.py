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
                num_votes = 1
                total_correct = 0
                total_seen = 0
                for fn in range(len(TEST_FILES)):
                    print('----' + str(fn) + '----')
                    current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
                    current_data = current_data[:, 0:NUM_POINT, :]
                    # current_data = np.expand_dims(current_data, axis=-2)
                    current_label = np.squeeze(current_label)
                    print(current_data.shape)

                    file_size = current_data.shape[0]
                    # num_batches = file_size // BATCH_SIZE
                    # print(file_size)


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
                                # print(Session_out, np.argmax(Session_out), label)
                                pred = np.argmax(Session_out)

                                # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
                                total_correct += 1 if pred == label else 0
                                total_seen += 1

                print('eval accuracy: %f' % (total_correct / float(total_seen)))
                    # print('eval avg class acc: %f' % (
                    #     np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))


def single_infer(pb_file, input_node):
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

                    data = current_data[0:1, :, :]
                    label = current_label[0]
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

                    nodes = {}
                    for ind, op in enumerate(graph.get_operations()):
                        # INFERENCE Here
                        # print([i for i in op.inputs], op.outputs,)
                        # print(op.type)
                        feed_dict = {}
                        if ind == 0:
                            init_input = graph.get_tensor_by_name(input_node)
                            nodes[input_node] = data
                        for inp in op.inputs:
                            # print(inp)
                            t = graph.get_tensor_by_name(inp.name)
                            # print(t.op.type)
                            if t.op.type != "Const" and t.op.type != 'Range':
                                feed_dict[t] = nodes[t.name]

                            # else:
                            #     if "mul_fold" in t.name.lower():
                            #         w = sess.run(t)
                            #         if len(w.shape) == 4:
                            #             w = w.transpose([3, 0, 1, 2])
                            #         print(t.name, np.max(w), np.min(w), w.shape, w.flatten()[:100])
                        # print(op.outputs)
                        if not op.outputs:
                            continue
                        l_outputs = []
                        for output in op.outputs:
                            l_outputs.append(graph.get_tensor_by_name(output.name))  # Output Tensor

                        if len(op.inputs) > 0:
                            # print(feed_dict.keys())
                            # if ind == len(ops) - 1:
                            #   feed_dict.setdefault(init_input, [image])
                            Session_out = sess.run(l_outputs, feed_dict=feed_dict)
                            # Session_out1 = sess.run(l_output, feed_dict={init_input: [image]})
                            # print(Session_out)
                            for ind, output in enumerate(op.outputs):
                                fea = Session_out[ind]
                                print(output.name, np.mean(fea), np.std(fea), np.sum(fea))
                                # if 'quant' in output.name:
                                #     print(np.max(fea), np.min(fea))
                                # if len(fea.flatten()) == 9:
                                #     print(fea)
                                if output.name == 'DGCNN/Reshape:0':
                                    print(fea, np.argmax(fea))
                                if output.name == 'DGCNN/dgcnn1/Conv/act_quant/FakeQuantWithMinMaxVars:0'\
                                    or output.name == 'DGCNN/dgcnn1/Max:0' \
                                    or output.name == 'DGCNN/get_edge_feature/concat_quant/FakeQuantWithMinMaxVars:0':
                                    # print(fea.flatten()[:200], fea.shape)
                                    print('\n'.join(map(str, (fea[0, 0, :20, :20]))))
                                if output.name == 'DGCNN/pairwise_distance/MatMul_quant/FakeQuantWithMinMaxVars:0'\
                                    or output.name == 'DGCNN/pairwise_distance/mul_quant/FakeQuantWithMinMaxVars:0'\
                                    or output.name == 'DGCNN/pairwise_distance/Mul_1_quant/FakeQuantWithMinMaxVars:0'\
                                    or output.name == 'DGCNN/pairwise_distance/Sum_quant/FakeQuantWithMinMaxVars:0'\
                                    or output.name == 'DGCNN/pairwise_distance/sub_quant/FakeQuantWithMinMaxVars:0' \
                                    or output.name == 'DGCNN/pairwise_distance/sub_1_quant/FakeQuantWithMinMaxVars:0' \
                                    or output.name == 'DGCNN/knn/TopKV2:0' \
                                    or output.name == 'DGCNN/knn/TopKV2:1':
                                    print(fea[0, :20, :20])
                                # if output.name == 'DGCNN/get_edge_feature_1/Tile:0' \
                                #     or output.name == 'DGCNN/get_edge_feature_1/GatherV2:0' \
                                #     or output.name == 'DGCNN/get_edge_feature_1/sub_quant/FakeQuantWithMinMaxVars:0':
                                #     print(fea.flatten()[:200])

                                if output not in nodes:
                                    nodes[output.name] = Session_out[ind]

                    # for name, tensor in nodes.items():
                    #     print(name, np.mean(tensor), np.std(tensor))


# show_graph("/media/wangzi/wangzi/codes/my_dgcnn/log_0828_best_quant_ori/dgcnn_quant.pb")
# infer_graph("/media/wangzi/wangzi/codes/my_dgcnn/log/dgcnn.pb", 'input:0', 'DGCNN/Reshape:0')
single_infer("/media/wangzi/wangzi/codes/my_dgcnn/log/dgcnn.pb", 'input:0')
