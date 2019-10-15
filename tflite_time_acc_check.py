import numpy as np
import tensorflow as tf
import os
import sys
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider


def find_neighbor(point_cloud):
    point_cloud = np.array(point_cloud)
    point_cloud_transpose = point_cloud.transpose([0, 2, 1])
    point_cloud_inner = np.matmul(point_cloud, point_cloud_transpose)
    # point_cloud_inner = -2 * point_cloud_inner
    point_cloud_inner = -2 * point_cloud_inner
    # point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keep_dims=True)
    point_cloud_square = np.sum(np.multiply(point_cloud, point_cloud), axis=-1, keepdims=True)
    point_cloud_square_transpose = np.transpose(point_cloud_square, axes=[0, 2, 1])
    # point_cloud_square_tranpose = tf.reduce_sum(tf.square(point_cloud_transpose), axis=-2, keep_dims=True)
    # return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    return point_cloud_inner + point_cloud_square + point_cloud_square_transpose


def naive_arg_topK(matrix, K, axis=0):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(matrix, axis=axis)  # ascending sort
    return full_sort.take(np.arange(K), axis=axis)


def tflite_infer(model):
    TEST_FILES = provider.getDataFiles(\
        os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
    NUM_POINT = 1024
    NUM_CLASSES = 40
    # output_file = "/home/wangzi/quant_log.txt"
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    all_details = interpreter.get_tensor_details()

    # print(input_details)
    # print(output_details)
    # print('\n'.join(map(str, all_details)))

    # import sys
    #
    # print(all_details[268])
    # print(interpreter.get_tensor(268))
    # sys.exit()
    # Test model on random input data.
    input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    num_votes = 1
    total_correct = 0
    total_seen = 0
    total_time = 0

    for fn in range(len(TEST_FILES)):
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_POINT, :]

        # current_data = np.round(current_data * 128 + 127)
        # current_data = current_data.astype(np.uint8)
        current_label = np.squeeze(current_label)
        # print(current_data.shape)

        file_size = current_data.shape[0]
        # print(file_size)

        for f_idx in range(file_size):
            for vote_idx in range(num_votes):
                # rotated_data = provider.rotate_point_cloud_by_angle(current_data[f_idx:f_idx+1, :, :],
                #                                                     vote_idx / float(num_votes) * np.pi * 2)

                data = current_data[f_idx:f_idx+1, :, :]
                feat = find_neighbor(data)
                knn = naive_arg_topK(feat, 20, -1)

                if "uint8" in model:
                    data = np.round(data * 128 + 127)
                    data = data.astype(np.uint8)
                knn = knn.astype(np.int32)

                # interpreter.set_tensor(input_details[0]['index'], current_data[f_idx:f_idx+1, :, :])
                interpreter.set_tensor(input_details[0]['index'], data)
                interpreter.set_tensor(input_details[1]['index'], knn)

                start = time.time()
                interpreter.invoke()
                end = time.time()
                total_time += end - start
                output_data = interpreter.get_tensor(output_details[0]['index'])
                pred = np.argmax(output_data)
                print(output_data, pred, current_label[f_idx])

                l = current_label[f_idx]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred == l)

                # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
                total_correct += 1 if pred == l else 0
                total_seen += 1

    print('eval accuracy: %f' % (total_correct / float(total_seen)))
    print('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    print("eval finish in {}".format(total_time))

# img = cv2.imread("/home/wangzi/Downloads/cifar_10_test/45daebfcdfea1599761e15998da6fbb8.jpg")
# input_data = cv2.resize(img, (input_shape[1], input_shape[2]))
# input_data = input_data.reshape(input_shape)
# interpreter.set_tensor(input_details[0]['index'], input_data)
#
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data, np.argmax(output_data))


if __name__ == "__main__":
    # start = time.time()
    tflite_infer("/media/wangzi/wangzi/codes/my_dgcnn/log_1010_quant_noD/dgcnn_float32.tflite")
    # end = time.time()
    # print("infer finish in {}".format(end - start))
