import os
import sys
import tensorflow as tf  # Default graph is initialized when the library is imported
import numpy as np
from math import floor
from tensorflow.python.platform import gfile
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import provider
import pc_util

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, '..', 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, '..', 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
NUM_POINT = 1024
NUM_CLASSES = 40
DUMP_DIR = '../dump'
BATCH_SIZE = 1
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, '..', 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]


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
                    h5f = os.path.join(BASE_DIR, "..", TEST_FILES[fn])
                    current_data, current_label = provider.loadDataFile(h5f)
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


def infer_graph2getnodes(pb_file, test_files, input_node, output_nodes):
    if not isinstance(output_nodes, list):
        output_nodes = [output_nodes]
    with tf.Graph().as_default() as graph:  # Set default graph as graph
        with tf.Session() as sess:
            # Load the graph in graph_def
            # We load the protobuf file from the disk and parse it to retrive the unserialized graph_drf
            with gfile.FastGFile(pb_file, 'rb') as f:
                # Read the image & get statstics
                current_data, current_label = provider.loadDataFile(TEST_FILES[test_files[0]])
                current_data = current_data[test_files[1]:test_files[1] + 1, 0:NUM_POINT, :]
                current_label = np.squeeze(current_label)
                label = current_label[test_files[1]]

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

                    Session_out = sess.run(l_output, feed_dict={l_input: current_data})
                    res.append(Session_out)
                return res


def get_weigth_quant(graph_file, input_file, initial_node, target_node, mode="MANUAL"):
    max_v, min_v = (None, None)
    if mode is "MANUAL":
        w = infer_graph2getnodes(graph_file, input_file, initial_node, target_node)
        max_v = np.max(w)
        min_v = np.min(w)
    if mode is "SAVED":
        max_v, min_v = infer_graph2getnodes(graph_file, input_file, initial_node,
                                   [target_node.replace('FakeQuantWithMinMaxVars', 'max/read'),
                                    target_node.replace('FakeQuantWithMinMaxVars', 'min/read')])
    # scale = (max_v - min_v) / 255
    # zero_point = round(-min_v / scale)
    zero_point = round(-255 * min_v / (max_v - min_v))
    scale = -min_v / zero_point
    return scale, int(zero_point)


def get_quantization(graph_file, input_file, initial_node, target_node, prev_node=None):
    if 'act_quant' in target_node or 'conv_quant' in target_node or 'bypass_quant' in target_node:
        max_v, min_v = infer_graph2getnodes(graph_file, input_file, initial_node,
                                   [target_node.replace('FakeQuantWithMinMaxVars', 'max/read'),
                                    target_node.replace('FakeQuantWithMinMaxVars', 'min/read')])
    elif 'input' in target_node:
        return 1. / 127, 128
    elif prev_node is not None:
        max_v, min_v = infer_graph2getnodes(graph_file, input_file, initial_node,
                                   [prev_node.replace('FakeQuantWithMinMaxVars', 'max/read'),
                                    prev_node.replace('FakeQuantWithMinMaxVars', 'min/read')])
    else:
        return None
    scale = (max_v - min_v) / 255
    zero_point = -min_v / scale
    return scale, int(zero_point)


def get_conv(graph_file, input_file, initial_node, input_node, weight_node, bias_node, res_node,
             log_file, stride, prev_node=None, double_check=True):
    inp, weight, res, bias = infer_graph2getnodes(graph_file, input_file, initial_node,
                                         [input_node, weight_node, res_node, bias_node])

    file_path = log_file
    inp_q = get_quantization(graph_file, input_file, initial_node, input_node, prev_node)
    # weight_q = get_quantization(graph_file, input_file, initial_node, weight_node)
    weight_q = get_weigth_quant(graph_file, input_file, initial_node, weight_node)
    print(weight_q)
    weight_dequant = weight / weight_q[0] + weight_q[1]
    if np.max(np.abs(weight_dequant - np.round(weight_dequant))) > .001:
        print(weight_dequant)
        weight_q = get_weigth_quant(graph_file, input_file, initial_node, weight_node, "SAVED")
        print(weight_q)
        weight_dequant = weight / weight_q[0] + weight_q[1]
    if np.max(np.abs(weight_dequant - np.round(weight_dequant))) > .001:
        print("warning weight is not totally quantized into uint8: ", weight_dequant)
        # raise Exception("Error")
    res_q = get_quantization(graph_file, input_file, initial_node, res_node)
    bias_q = (inp_q[0] * weight_q[0], 0)
    stride_h, stride_w = stride
    # igs1, out1 = inferer_graph("/media/wangzi/wangzi/models/train_logs_inceptionv1_q/inception_v1.pb",
    #               '/home/wangzi/Desktop/deer7.png', 'input:0', 'InceptionV1/InceptionV1/MaxPool_2a_3x3/MaxPool:0')
    # print("input:")
    # print('\n'.join(map(str, ['\t'.join(map(str, x)) for x in out[0, 0:10, 0:10, 0]])))
    # print("output:")
    # print('\n'.join(map(str, ['\t'.join(map(str, x)) for x in out1[0, 0:10, 0:10, 0]])))
    # inputs = ([i for i in igs])
    # labels = [np.argmax(n) for n in out]
    # print(list(zip(inputs, labels)))

    with open(file_path, "w") as f_:
        _, inp_h, inp_w, inp_c = inp.shape
        _, res_h, res_w, res_c = res.shape
        k_h, k_w, _, _ = weight.shape
        pad_h = (res_h - 1) * stride_h - inp_h + k_h
        pad_w = (res_w - 1) * stride_w - inp_w + k_w
        f_.write(
            'input_s {} input_z {} filter_s {} filter_z {} output_s {} output_z {} input_width {} input_height {} '
            'input_depth {} output_width {} output_height {} output_depth {} filter_width {} filter_height {} '
            'pad_width {} pad_height {} stride_width {} stride_height {}\n'.
                format(inp_q[0], inp_q[1], weight_q[0], weight_q[1], res_q[0], res_q[1], inp_w, inp_h,
                       inp_c, res_w, res_h, res_c, k_w, k_h, int(floor(pad_w / 2)), int(floor(pad_h / 2)), stride_w,
                       stride_h))
        inp = inp / inp_q[0] + inp_q[1]
        if double_check:
            print("Input ", inp)
        inp = np.round(inp)
        inp = inp.astype(np.int)
        f_.write("Input: ")
        for b in range(inp.shape[0]):
            for h in range(inp.shape[1]):
                for w in range(inp.shape[2]):
                    for c in range(inp.shape[3]):
                        f_.write(str(inp[b][h][w][c]) + " ")
        f_.write('\n')

        f_.write("Weight: ")
        weight = weight / weight_q[0] + weight_q[1]
        if double_check:
            print("Weight ", weight)
            print(weight.shape)
        weight = np.round(weight)
        weight = weight.astype(np.int)
        for h in range(weight.shape[0]):
            for w in range(weight.shape[1]):
                for b in range(weight.shape[2]):
                    for c in range(weight.shape[3]):
                        f_.write(str(weight[h][w][b][c]) + " ")
        f_.write('\n')

        f_.write("Bias: ")
        print("ori bias", bias)
        bias = bias / bias_q[0] + bias_q[1]
        if double_check:
            print("bias ", bias)
            print(bias.shape)
        bias = np.squeeze(bias)
        bias = np.round(bias)
        bias = bias.astype(np.int)
        for c in range(bias.shape[0]):
            f_.write(str(bias[c]) + " ")
        f_.write('\n')

        f_.write("Result: ")
        res = res / res_q[0] + res_q[1]
        if double_check:
            print("Res ", res, " quant ", res_q)
        res = np.round(res)
        res = res.astype(np.int)
        for b in range(res.shape[0]):
            for h in range(res.shape[1]):
                for w in range(res.shape[2]):
                    for c in range(res.shape[3]):
                        f_.write(str(res[b][h][w][c]) + " ")
        f_.write('\n')
        print("file is written in {}".format(log_file))


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
                    # print(data)

                    # data = np.round(data * 128 + 127)  # quant to [0, 255]
                    # data = (data - 127) / 128

                    # print(data)
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
                                print(output.name, fea.shape, np.mean(fea), np.std(fea), np.sum(fea))
                                # if 'quant' in output.name:
                                #     print(np.max(fea), np.min(fea))
                                # if len(fea.flatten()) == 9:
                                #     print(fea)
                                if output.name == 'DGCNN/Reshape:0' or output.name == 'PointNet/Reshape:0':
                                    print(fea, np.argmax(fea), label)
                                if output.name == 'DGCNN/dgcnn1/Conv/act_quant/FakeQuantWithMinMaxVars:0'\
                                    or output.name == 'DGCNN/dgcnn1/Max:0' \
                                    or output.name == 'DGCNN/get_edge_feature/concat_quant/FakeQuantWithMinMaxVars:0':
                                    # print(fea.flatten()[:200], fea.shape)
                                    print('\n'.join(map(str, (fea[0, 0, :20, :20]))))
                                if output.name in ['DGCNN/pairwise_distance/MatMul_quant/FakeQuantWithMinMaxVars:0',
                                     'DGCNN/pairwise_distance/mul_quant/FakeQuantWithMinMaxVars:0',
                                     'DGCNN/pairwise_distance/Mul_1_quant/FakeQuantWithMinMaxVars:0',
                                     'DGCNN/pairwise_distance/Sum_quant/FakeQuantWithMinMaxVars:0',
                                     'DGCNN/pairwise_distance/sub_quant/FakeQuantWithMinMaxVars:0',
                                     'DGCNN/pairwise_distance/sub_1_quant/FakeQuantWithMinMaxVars:0',
                                    # 'DGCNN/knn/TopKV2:0',
                                    'DGCNN/knn/TopKV2:1',
                                    'DGCNN/get_edge_feature/GatherV2:0',
                                    'DGCNN/transform_net/tconv1/act_quant/FakeQuantWithMinMaxVars:0',
                                    'DGCNN/transform_net/transform_XYZ/act_quant/FakeQuantWithMinMaxVars:0',
                                    'DGCNN/Transform/MatMul_quant/FakeQuantWithMinMaxVars:0',
                                    # 'DGCNN/transform_net/tconv1/weights/read:0',

                                    ]:
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
# infer_graph("/media/wangzi/wangzi/codes/my_dgcnn/log_0929_quantAfterTopK_noSTN_noD/dgcnn.pb", 'input:0', 'DGCNN/Reshape:0')
single_infer("/media/wangzi/wangzi/codes/my_dgcnn/logs/log_1106_.75_quant_noD/dgcnn.pb", 'input:0')
# get_conv(graph_file="/media/wangzi/wangzi/codes/my_dgcnn/log_1129_pointnet512_quant/pointnet.pb", input_file=[0,0],
#          initial_node='input:0', input_node='PointNet/transform_net1/transform_net/tconv2/act_quant/FakeQuantWithMinMaxVars:0',
#          weight_node='PointNet/transform_net1/transform_net/tconv3/weights_quant/FakeQuantWithMinMaxVars:0',
#          bias_node='PointNet/transform_net1/transform_net/tconv3/BatchNorm_Fold/bias:0',
#          res_node='PointNet/transform_net1/transform_net/tconv3/act_quant/FakeQuantWithMinMaxVars:0',
#              log_file='/media/wangzi/wangzi/codes/my_dgcnn/log_1129_pointnet512_quant/1x1conv_d1024', stride=[1, 1],
#          # prev_node='DGCNN/transform_net/tconv2/act_quant/FakeQuantWithMinMaxVars:0',
#          double_check=True)