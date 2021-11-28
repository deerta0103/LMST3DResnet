from __future__ import print_function
import os
import sys
import pickle
import time
import numpy as np
import h5py
import math

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import deepst.metrics as metrics
from deepst.datasets import BikeNYC
from deepst.model import lmst3d_resnet_nyc
from deepst.evaluation import evaluate

# np.random.seed(1337)  # for reproducibility

# parameters
DATAPATH = './data'
CACHEDATA = True  # cache data or NOT
path_cache = os.path.join(DATAPATH, 'CACHE', 'lmst3d_resnet')  # cache path
nb_epoch = 1000  # number of epoch at training stage
nb_epoch_cont = 20 # number of epoch at training (cont) stage
batch_size = 32  # batch size
T = 24  # number of time intervals in one day
lr = 0.0002  # learning rate
len_closeness = 3  # length of closeness dependent sequence - should be 6
len_period = 3  # length of peroid dependent sequence
len_trend = 3  # length of trend dependent sequence

nb_flow = 2  # there are two types of flows: inflow and outflow

# divide data into two subsets: Train & Test, of which the test set is the
# last 10 days
days_test = 10
len_test = T * days_test
map_height, map_width = 16, 8  # grid size
# For NYC Bike data, there are 81 available grid-based areas, each of
# which includes at least ONE bike station. Therefore, we modify the final
# RMSE by multiplying the following factor (i.e., factor).
nb_area = 81
m_factor = math.sqrt(1. * map_height * map_width / nb_area)

path_result = 'RET'
path_model = 'MODEL'
if os.path.isdir(path_result) is False:
    os.mkdir(path_result)
if os.path.isdir(path_model) is False:
    os.mkdir(path_model)
if CACHEDATA and os.path.isdir(path_cache) is False:
    os.mkdir(path_cache)

def build_model(save_model_pic=False):
    model = lmst3d_resnet(len_closeness, len_period, len_trend, nb_flow, map_height, map_width, external_dim)
    adam = Adam(lr=lr)
    model.compile(loss='mse', optimizer=adam, metrics=[metrics.rmse])
    model.summary()
    if (save_model_pic):
        from keras.utils.vis_utils import plot_model
        plot_model(model, to_file='BikeNYC_model.png', show_shapes=True)
    return model

def read_cache(fname):
    mmn = pickle.load(open('preprocessing_bikenyc.pkl', 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test

def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()

# main code
# load data
print("loading data...")
ts = time.time()
fname = os.path.join(path_cache, 'BikeNYC_C{}_P{}_T{}.h5'.format(
    len_closeness, len_period, len_trend))
# if os.path.exists(fname) and CACHEDATA:
#     X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
#         fname)
#     print("load %s successfully" % fname)
# else:
X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = BikeNYC.load_data(
        T=T, nb_flow=nb_flow, len_closeness=len_closeness, len_period=len_period, len_trend=len_trend, len_test=len_test,
        preprocess_name='preprocessing_bikenyc.pkl', meta_data=True, datapath=DATAPATH)
if CACHEDATA:
        cache(fname, X_train, Y_train, X_test, Y_test,
              external_dim, timestamp_train, timestamp_test)

print("\n days (test): ", [v[:8] for v in timestamp_test[0::T]])
print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

print('=' * 10)

# training-test-evaluation iterations
for i in range(0,1):
    print('=' * 10)
    print("compiling model...")

    # lr_callback = LearningRateScheduler(lrschedule)

    # build model
    model = build_model(save_model_pic=False)

    hyperparams_name = 'BikeNYC.c{}.p{}.t{}.iter{}'.format(
        len_closeness, len_period, len_trend, i)
    fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))
    print(hyperparams_name)

    early_stopping = EarlyStopping(monitor='val_rmse', patience=25, mode='min')
    model_checkpoint = ModelCheckpoint(
        fname_param, monitor='val_rmse', verbose=0, save_best_only=True, mode='min')

    print('=' * 10)
    # train model
    np.random.seed(i*18)
    tf.set_random_seed(i*18)
    # tf.random.set_seed(i*18)
    print("training model...")
    history = model.fit(X_train, Y_train,
                        epochs=5,
                        batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=2)
    model.save_weights(os.path.join(
        path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
    model.save(os.path.join(
        path_model, '{}123huizong.h5'.format(hyperparams_name)), overwrite=True)

    pickle.dump((history.history), open(os.path.join(
        path_result, '{}.history.pkl'.format(hyperparams_name)), 'wb'))

    print('=' * 10)

    # evaluate model
    print('evaluating using the model that has the best loss on the valid set')
    model.load_weights(fname_param) # load best weights for current iteration
    
    Y_pred = model.predict(X_test) # compute predictions
    score = model.evaluate(X_train, Y_train, batch_size=Y_train.shape[0] // 48, verbose=0)
    print('Train score: %.6f  rmse (real): %.6f' % (score[0], score[1] * 1.58))
    test_start_time = time.time()
    score = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=0)
    test_end_time = time.time()
    print("test time(s): %.4f s" % (test_end_time - test_start_time))
    print('Test score: %.6f  rmse (real): %.6f' % (score[0], score[1] * 1.58))

    # score = evaluate(Y_test, Y_pred, mmn, rmse_factor=1) # evaluate performance

   
    K.clear_session()