import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model, Model
import scipy.io as sio
import data_preprocess
from keras.models import Sequential, load_model
from keras.layers import * #Dense, LSTM, Dropout, GRU, Conv2D, MaxPooling2D, Flatten, merge, Merge, UpSampling2D, \

which_data = "cifar100"  # dataset name
test_size = 10   # should NOT be modified

# path configuration
load_path = "/opt/Data/tsl/ResNetHash/saved/2017101802/cifar100/su_ae/data/hash_su_ae.h5"
hash_file_path = "./"+which_data+"_hash_32bits_res_testset_su_ae.mat"

hash_bits = 32  # bit length for hash codes (must be corresponding to the trained model)

hash_output = None
if __name__ == '__main__':
    # load data
    (x_train, y_train), (x_test, y_test) = load_data.load_data(which_data, False)

    resnet = load_model(load_path, compile=False)
    print resnet.summary()
    print "hash code length is:", resnet.get_layer("hash_x").output.get_shape().as_list()[-1]
    hash_model = Model(input=resnet.input,
                       output=resnet.get_layer("hash_x").output)

    batches = len(x_test) / test_size

    print "generate hash now ..."
    for i in range(batches):
        x_batch = x_test[i*test_size:(i+1)*test_size, :, :, :]
        y_batch = y_test[i*test_size:(i+1)*test_size]
        hash_temp = hash_model.predict(x_batch)
        hash_temp = np.array(hash_temp, float)
        hash_temp = np.reshape(hash_temp, [test_size, hash_bits])

        if i == 0:
            hash_output = hash_temp
        else:
            hash_output = np.concatenate((hash_output, hash_temp))
    print hash_output.shape
    hash_binary = np.where(hash_output > 0.5, 1, 0)

    print "save hash to file ..."+hash_file_path
    sio.savemat(hash_file_path, {"hash_org": hash_output, "hash_bin": hash_binary, "label": y_test})