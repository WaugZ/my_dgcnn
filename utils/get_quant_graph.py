import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

from official_utils import *


def get_quant_graph_def(input_graph_def, mins, maxes):
    input_node_map = {}  # key: node.name, value: node(containing much information)
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for ", node.name)

    # print(input_node_map)
    nodes_to_skip = {}
    new_ops = []

    # print([(node.name, node.input, node.op) for node in input_graph_def.node])
    for node in input_graph_def.node:  # weights_quant
        if node.op not in ('DepthwiseConv2dNative', 'Conv2D'):
            continue
        nodes_to_skip[node.name] = True

        weight_op = node_from_map(input_node_map, node.input[1])
        weight = values_from_const(weight_op)
        # print(weight_op.attr["dtype"])

        quant_name_prefix = weight_op.name.replace(weight_op.name.split('/')[-1], "weights_quant/")

        max_op = node_def_pb2.NodeDef()
        max_op.op = 'Const'
        max_op.name = quant_name_prefix + "max"
        max_op.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        max_op.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            [np.max(weight)], dtypes.float32, [])))
        # print(max_op.attr["value"])

        min_op = node_def_pb2.NodeDef()
        min_op.name = quant_name_prefix + "min"
        min_op.op = 'Const'
        min_op.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
        min_op.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            [np.min(weight)], dtypes.float32, [])))

        quant_op = node_def_pb2.NodeDef()
        quant_op.name = quant_name_prefix + "FakeQuantWithMinMaxVars"
        quant_op.op = "FakeQuantWithMinMaxVars"
        quant_op.attr["narrow_range"].CopyFrom(attr_value_pb2.AttrValue(b=True))
        quant_op.attr["num_bits"].CopyFrom(attr_value_pb2.AttrValue(i=8))
        quant_op.input.extend([weight_op.name, min_op.name, max_op.name])  # the input order must be weight, min, max
        # print(quant_op)

        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        new_node.input[-1] = quant_op.name
        new_ops.extend([max_op, min_op, quant_op, new_node])
        # print(node.name, input_node_map[node.input[1]].name)

    weights_quant_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip: continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        weights_quant_graph_def.node.extend([new_node])

    weights_quant_graph_def.node.extend(new_ops)
    # f = gfile.FastGFile(out_pb, 'w')
    # f.write(weights_quant_graph_def.SerializeToString())

    input_node_map = {}  # key: node.name, value: node(containing much information)
    for node in weights_quant_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for ", node.name)

    new_ops = []
    nodes_to_skip = {}

    for node in weights_quant_graph_def.node:  # act_quant: activation layer and passby layer
        activate_layer_flag = False
        node_inputs = []  # ori_inputs
        modify_inputs = []
        input_node_to_skip = {}
        for inp in node.input:
            node_input = node_from_map(input_node_map, inp)
            node_inputs.append(node_input)
            if node_input.op in ('Relu6', 'Relu', 'Add', 'BiasAdd'):  # activation layer and passby layer
                activate_layer_flag = True  # todo: other relu like relu1
        if not activate_layer_flag: continue
        if node.op in ('Relu6', 'Relu') and node_from_map(input_node_map, inp).op in ('Add', 'BiasAdd'):
            continue  # this layer is relu and in the case there will only be 1 input

        # print(node.name, [n.name for n in node_inputs])
        nodes_to_skip[node.name] = True

        for node_input in node_inputs:
            if node_input.op not in ('Relu6', 'Relu', 'Add', 'BiasAdd'):  # activation layer
                continue
            quant_name_prefix = node_input.name.replace(node_input.name.split('/')[-1], "act_quant/")
            # print(node_input.name, node_input.name + ":0" in maxes)

            max_op = node_def_pb2.NodeDef()
            max_op.op = 'Const'
            max_op.name = quant_name_prefix + "max"
            max_op.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            max_op.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                [maxes[node_input.name + ":0"]], dtypes.float32, [])))
            # print(max_op.attr["value"])

            min_op = node_def_pb2.NodeDef()
            min_op.name = quant_name_prefix + "min"
            min_op.op = 'Const'
            min_op.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=dtypes.float32.as_datatype_enum))
            min_op.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
                [mins[node_input.name + ":0"]], dtypes.float32, [])))

            quant_op = node_def_pb2.NodeDef()
            quant_op.name = quant_name_prefix + "FakeQuantWithMinMaxVars"
            quant_op.op = "FakeQuantWithMinMaxVars"
            quant_op.attr["narrow_range"].CopyFrom(attr_value_pb2.AttrValue(b=True))
            quant_op.attr["num_bits"].CopyFrom(attr_value_pb2.AttrValue(i=8))
            quant_op.input.extend([node_input.name, min_op.name, max_op.name]) # the input order must be act, min, max

            if quant_op.name not in [op.name for op in new_ops]:
                new_ops.extend([max_op, min_op, quant_op])
            modify_inputs.append(quant_op.name)
            input_node_to_skip[node_input.name] = True
            # print(quant_op)

        new_op = node_def_pb2.NodeDef()
        new_op.CopyFrom(node)
        del new_op.input[:]
        new_op.input.extend(modify_inputs)
        for ori_input in node_inputs:
            if ori_input.name in input_node_to_skip:
                continue
            new_op.input.extend([ori_input.name])

        new_ops.extend([new_op])
        # print(new_op)

    # print(nodes_to_skip)
    act_quant_graph_def = graph_pb2.GraphDef()
    for node in weights_quant_graph_def.node:
        if node.name in nodes_to_skip: continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        act_quant_graph_def.node.extend([new_node])
        # print(new_node)
    act_quant_graph_def.node.extend(new_ops)
    # print(new_ops)
    # f = gfile.FastGFile(out_pb, 'w')
    # f.write(act_quant_graph_def.SerializeToString())
    return act_quant_graph_def