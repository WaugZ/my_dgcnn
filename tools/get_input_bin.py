import numpy as np
import tensorflow as tf
import os
import sys
import struct
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, '..', 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
NUM_POINT = 1024
NUM_CLASSES = 40

for fn in range(len(TEST_FILES)):
    h5f = os.path.join(BASE_DIR, '..', TEST_FILES[fn])
    current_data, current_label = provider.loadDataFile(h5f)
    current_data = current_data[:, 0:NUM_POINT, :]
    # current_data = np.expand_dims(current_data, axis=-2)
    current_label = np.squeeze(current_label)
    # print(current_data.shape)

    file_size = current_data.shape[0]
    # print(file_size)
    print(current_data, current_data.shape)
    for ind in range(file_size):
        f_data = current_data[ind:ind+1, :, :]
        f_data_flatten = f_data.flatten()
        flatten_size = len(f_data_flatten)
        f_label = current_label[ind]
        print(f_data, f_label)
        bin_data = struct.pack("f" * flatten_size, *f_data_flatten)
        if not os.path.exists("data_bin"):
            os.mkdir("data_bin")
        with open("data_bin/{}_{}_{}_{}.bin".format(fn, ind, NUM_POINT, f_label), 'wb') as f:
            f.write(bin_data)
