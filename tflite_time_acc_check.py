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

    for fn in range(len(TEST_FILES)):
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_POINT, :]
        current_data = np.round(current_data * 128 + 127)
        current_data = current_data.astype(np.uint8)
        # current_data = np.expand_dims(current_data, axis=-2)
        current_label = np.squeeze(current_label)
        # print(current_data.shape)

        file_size = current_data.shape[0]
        # print(file_size)

        for f_idx in range(file_size):
            for vote_idx in range(num_votes):
                # rotated_data = provider.rotate_point_cloud_by_angle(current_data[f_idx:f_idx+1, :, :],
                #                                                     vote_idx / float(num_votes) * np.pi * 2)

                data = current_data[f_idx:f_idx+1, :, :]
                # interpreter.set_tensor(input_details[0]['index'], current_data[f_idx:f_idx+1, :, :])
                interpreter.set_tensor(input_details[0]['index'], data)

                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                pred = np.argmax(output_data)
                # print(output_data, pred, current_label[f_idx])

                l = current_label[f_idx]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred == l)

                # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
                total_correct += 1 if pred == l else 0
                total_seen += 1

    print('eval accuracy: %f' % (total_correct / float(total_seen)))
    print('eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
# img = cv2.imread("/home/wangzi/Downloads/cifar_10_test/45daebfcdfea1599761e15998da6fbb8.jpg")
# input_data = cv2.resize(img, (input_shape[1], input_shape[2]))
# input_data = input_data.reshape(input_shape)
# interpreter.set_tensor(input_details[0]['index'], input_data)
#
# interpreter.invoke()
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data, np.argmax(output_data))


if __name__ == "__main__":
    start = time.time()
    tflite_infer("/media/wangzi/wangzi/codes/my_dgcnn/log_0919_quant_noSTN_noD_relu6/dgcnn_uint8.tflite")
    end = time.time()
    print("infer finish in {}".format(end - start))
