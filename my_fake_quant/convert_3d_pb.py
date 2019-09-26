from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import app

from quant_all_nodes import quant_normal_op
from forward2get_min_max import forward_batch2get_min_max, forward2get_min_max

read_pb = "/media/wangzi/wangzi/codes/my_dgcnn/log_0919_quant_noSTN_noD_relu6/dgcnn.pb"
out_pb = "/media/wangzi/wangzi/codes/my_dgcnn/log_0919_quant_noSTN_noD_relu6/dgcnn_quant.pb"
input_node = "input:0"
input_quant = (0, 1)  # data are range in [-1, 1] already
NUM = 188

if __name__ == "__main__":
    mins, maxes = forward2get_min_max(read_pb, input_node, input_quant, num=NUM)
    input_graph_def = graph_pb2.GraphDef()

    with gfile.Open(read_pb, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    op_quant_graph_def = quant_normal_op(input_graph_def, mins, maxes)
    with gfile.FastGFile(out_pb, 'w') as f:
        f.write(op_quant_graph_def.SerializeToString())
    print("transform finish.")
