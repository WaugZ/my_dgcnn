import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.python.platform import gfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import provider

TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
NUM_POINT = 1024
NUM_CLASSES = 40
BATCH_SIZE = 1


def forward_graph(pb_file, input_node, input_data, input_quant=(0., 1.), maxes={}, mins={}):
    """
        forward infer the graph, record the ever max or min value of the nodes need quantization
        """
    with tf.Graph().as_default() as graph:  # Set default graph as graph
        with tf.Session() as sess:
            # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
            with gfile.FastGFile(pb_file, 'rb') as f:
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
                quant_data = (input_data - input_quant[0]) / input_quant[1]  # input data in range [-1, 1] convert to [0, 1]
                # print(np.min(quant_data), np.max(quant_data))
                nodes = {}

                # initialize_all_variables
                tf.global_variables_initializer()
                init_input = None
                for ind, op in enumerate(graph.get_operations()):
                    # INFERENCE Here
                    # print([i for i in op.inputs], op.outputs,)
                    # print(op.type)
                    feed_dict = {}
                    if ind == 0:
                        init_input = graph.get_tensor_by_name(input_node)
                        nodes[input_node] = quant_data
                    for inp in op.inputs:
                        # print(inp)
                        t = graph.get_tensor_by_name(inp.name)
                        # print(t.op.type)
                        if t.op.type != "Const" and t.op.type != 'Range':
                            feed_dict[t] = nodes[t.name]
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
                            if output not in nodes:
                                nodes[output.name] = Session_out[ind]

                    # print(nodes.keys())

                print(np.argmax(nodes["DGCNN/Reshape:0"]))
                for name, tensor in nodes.items():
                    max_v = np.max(tensor)
                    min_v = np.min(tensor)
                    if name in maxes:
                        if maxes[name] < max_v:
                            maxes[name] = max_v
                    else:
                        maxes[name] = max_v
                    if name in mins:
                        if mins[name] > min_v:
                            mins[name] = min_v
                    else:
                        mins[name] = min_v


def forward_batch2get_min_max(pb_file, input_node, inp_quant, num=10):
    maxes = {}
    mins = {}

    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        print('----' + str(fn) + '----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_label = np.squeeze(current_label)
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        print(current_data.shape)

        file_size = current_data.shape[0]
        print(file_size)

        num_votes = 1
        for f_idx in range(file_size):
            if f_idx > num:
                break
            for vote_idx in range(num_votes):

                data = current_data[f_idx:f_idx + 1, :, :]
                label = current_label[f_idx]
                print(label, end=',')
                forward_graph(pb_file, input_node, data, inp_quant, maxes, mins)

    return mins, maxes


def forward2get_min_max(pb_file, input_node, inp_quant, num=10):
    """
            forward infer the graph, record the ever max or min value of the nodes need quantization
            """
    maxes = {}
    mins = {}
    with tf.Graph().as_default() as graph:  # Set default graph as graph
        with tf.Session() as sess:
            # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
            with gfile.FastGFile(pb_file, 'rb') as f:
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

                tf.global_variables_initializer()

                train_file_idxs = np.arange(0, len(TRAIN_FILES))
                np.random.shuffle(train_file_idxs)

                for fn in range(len(TRAIN_FILES)):
                    print('----' + str(fn) + '----')
                    current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
                    current_data = current_data[:, 0:NUM_POINT, :]
                    current_label = np.squeeze(current_label)
                    current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
                    print(current_data.shape)

                    file_size = current_data.shape[0]
                    print(file_size)

                    num_votes = 1
                    nodes = {}

                    for f_idx in range(file_size):
                        if f_idx > num:
                            break
                        for vote_idx in range(num_votes):
                            data = current_data[f_idx:f_idx + 1, :, :]
                            label = current_label[f_idx]
                            print(label, end=',')
                            # forward_graph(pb_file, input_node, data, inp_quant, maxes, mins)
                            quant_data = (data - inp_quant[0]) / inp_quant[1]  # input data in range [-1, 1] convert to [0, 1]
                            # print(np.min(quant_data), np.max(quant_data))

                            # initialize_all_variables
                            init_input = None
                            for ind, op in enumerate(graph.get_operations()):
                                # INFERENCE Here
                                # print([i for i in op.inputs], op.outputs,)
                                # print(op.type)
                                feed_dict = {}
                                if ind == 0:
                                    init_input = graph.get_tensor_by_name(input_node)
                                    nodes[input_node] = quant_data
                                for inp in op.inputs:
                                    # print(inp)
                                    t = graph.get_tensor_by_name(inp.name)
                                    # print(t.op.type)
                                    if t.op.type != "Const" and t.op.type != 'Range':
                                        feed_dict[t] = nodes[t.name]
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
                                        if output not in nodes:
                                            nodes[output.name] = Session_out[ind]

                                # print(nodes.keys())

                            print(np.argmax(nodes["DGCNN/Reshape:0"]))
                            for name, tensor in nodes.items():
                                max_v = np.max(tensor)
                                min_v = np.min(tensor)
                                if name in maxes:
                                    if maxes[name] < max_v:
                                        maxes[name] = max_v
                                else:
                                    maxes[name] = max_v
                                if name in mins:
                                    if mins[name] > min_v:
                                        mins[name] = min_v
                                else:
                                    mins[name] = min_v

    return mins, maxes


if __name__ == "__main__":
    # mins, maxes = forward_batch2get_min_max("/media/wangzi/wangzi/codes/my_dgcnn/log_0828_best_quant_ori/dgcnn_quant.pb", "input:0", (127.5, 127.5))
    mins, maxes = forward2get_min_max("/media/wangzi/wangzi/codes/my_dgcnn/log_0828_best_quant_ori/dgcnn.pb", "input:0", (0, 1))
    print("min ", mins)
    print("max ", maxes)
