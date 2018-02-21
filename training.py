import tensorflow as tf
import keras
from keras.datasets import cifar10, mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model, load_model
from keras import optimizers
import os
import numpy as np
import hash_model
import data_preprocess
from keras.layers import * #Dense, LSTM, Dropout, GRU, Conv2D, MaxPooling2D, Flatten, merge, Merge, UpSampling2D, \

which_data = "mnist"  
num_classes = 10       
stack_num = 18    # number of stack in the ResNet network
batch_size = 64   # number of training batch per step
epochs = 90       # number of training epoch

# weight in the loss function
alpha = 1e-1    # weight of binary loss term
beta = 1e-1     # weight of evenly distributed term
gamma = 1   # weight of recovery loss term

hash_bits = 32  # length of hash bits to encode


base = './saved/'+which_data+'/SAEH/'  # model and log path to be saved
load_path = ""
save_path = base + "data/"
log_path = base + "log/"

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

# changing weight scheduler, we change the learning weight according to the epoch
def scheduler(epoch):
    if epoch <= 30:
        return 0.1
    if epoch <= 55:
        return 0.02
    if epoch <= 75:
        return 0.004
    return 0.0008


# main function
if __name__ == '__main__':
    # load the dataset
    # if you want to
    (x_train, y_train), (x_test, y_test) = load_data.load_data(which_data)
    (_, img_rows, img_cols, img_channels) = x_train.shape

    # build our Supervised Auto-encoder Hashing network
    hash_su_ae_model = hash_model.HashSupervisedAutoEncoderModel(img_rows, img_cols, img_channels, num_classes, stack_num, hash_bits, alpha, beta, gamma)
    
    resnet = Model(inputs=hash_su_ae_model.img_input, outputs=[hash_su_ae_model.y_predict, hash_su_ae_model.y_decoded])
    if load_path is not None and load_path != "":
        resnet.load_weights(load_path)

    print resnet.summary()

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(optimizer=sgd,
                   loss={'y_predict': 'categorical_crossentropy',
                         'y_decoded': hash_su_ae_model.net_loss},
                   metrics={'y_predict': 'accuracy'},
                   loss_weights=[1, 1.],
                   )

    # set callback
    tb_cb = TensorBoard(log_dir=log_path, histogram_freq=0, write_graph=False)
    change_lr = LearningRateScheduler(scheduler)
    chk_pt = ModelCheckpoint(filepath=save_path+'chk_pt.h5', monitor='val_y_predict_acc', save_best_only=True, mode='max', period=1)
    cbks = [change_lr, tb_cb, chk_pt]

    resnet.fit(x_train, {"y_predict": y_train, "y_decoded":x_train}, epochs=epochs, batch_size=batch_size, callbacks=cbks, validation_data=(x_test, [y_test, x_test]))


    resnet.save(save_path+"hash_su_ae.h5")
    print "save model data at '", save_path+"hash_su_ae.h5"+"'"