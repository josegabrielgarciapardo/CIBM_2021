# When problems with CUDA libraries
# sudo ldconfig /usr/local/cuda-9.0/lib64/

import os
# USE THE SELECTED GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import warnings
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.preprocessing.image import save_img
from termcolor import colored
import tensorflow.keras.backend as K
from scipy.optimize import linear_sum_assignment


# GETTING REPRODUCIBLE THE RESULTS WITH KERAS
from numpy.random import seed
seed(1)
tf.random.set_seed(2)
#--------------------------------------------
warnings.filterwarnings('ignore')


# Load  data
def load_data(folders, data_opt, folder_csv):
    X_img = []
    for i in range(len(folder_csv.ID)):
        # Images
        img = cv2.imread(folders['images'] + folder_csv.ID[i])
        #cv2.imwrite(folders['general'] + '/images/' + folder_csv.ID[i], img)
        img = cv2.resize(img, (data_opt['y'], data_opt['x']))
        img = img / 255.0
        X_img.append(img)

    y = np.array(folder_csv.GT)
    X_img = np.array(X_img)

    return X_img, y

# Fit CAE
def fit_CAE(folders, X_train, params_DC, model, exp):

    # Hyper parameters
    learning_rate = params_DC['hyperparameters']['lr']
    loss_fun = params_DC['hyperparameters']['lf_CAE']
    optimizer = params_DC['hyperparameters']['opti']
    batchSize = params_DC['hyperparameters']['bs']
    numEpochs = params_DC['hyperparameters']['numEpochs']

    # Optimizer
    opt = optimizer(lr=learning_rate)
    # Name to save the optimizer
    opti_str = str(opt)
    opti_chain = opti_str.split('.')[-1]
    opti = opti_chain.split(' ')[0]

    name_to_save = 'Model_' + str(exp) + '_bs' + str(batchSize) + '_lr' + str(learning_rate) + '_' + loss_fun + '_' + \
                   opti + '_numConvs_' + str(params_DC['num_convs']) + '_firstFilter_' + str(params_DC['first_filter'])

    # Compile
    model.compile(optimizer=opt, loss=loss_fun)

    callBacks = [EarlyStopping(monitor='loss', patience=10, verbose=1, mode='auto'),
                 ModelCheckpoint(folders['models'] + 'CAE/' + name_to_save + '.hdf5',
                                monitor='loss', verbose=1, save_best_only=True, save_weights_only=False)]

    # Training with all options
    H = model.fit(x=X_train, y=X_train, batch_size=batchSize, epochs=numEpochs, verbose=1, callbacks=callBacks)

    return H, model, name_to_save

# Fit DC model
def fit_DC(folders, X_train, y_train, params_DC, model, exp):

    early_stopping = True

    # Hyper parameters
    learning_rate = params_DC['hyperparameters']['lr']
    loss_CAE = params_DC['hyperparameters']['lf_CAE']
    loss_DC = params_DC['hyperparameters']['lf_DC']
    optimizer = params_DC['hyperparameters']['opti']
    batchSize = params_DC['hyperparameters']['bs']
    numEpochs = params_DC['hyperparameters']['numEpochs']

    # Optimizer
    opt = optimizer(lr=learning_rate)
    # Name to save the optimizer
    opti_str = str(opt)
    opti_chain = opti_str.split('.')[-1]
    opti = opti_chain.split(' ')[0]

    name_to_save = 'Model_' + str(exp) + '_bs' + str(batchSize) + '_lr' + str(learning_rate) + '_' + opti + \
                   '_' + str(params_DC['num_convs']) + '_' + str(params_DC['first_filter']) + '_hybrid_' + str(params_DC['hybrid'])

    # Compile
    if params_DC['hybrid']==True:
        model.compile(optimizer=opt, loss=[loss_DC, loss_CAE], loss_weights=params_DC['weights'])
    else:
        model.compile(optimizer=opt, loss=loss_DC)

    index, patience, flag = 0,0,0
    best = float('inf')
    patience_max = 20
    for e in range(numEpochs):
        if params_DC['hybrid']:
            q, _ = model.predict(X_train)
        else:
            q = model.predict(X_train)
        p = target_distribution(q)

        while flag==0:
            # train on batch
            if (index + 1) * batchSize > X_train.shape[0]: # Last batch of each epoch
                loss = model.train_on_batch(x=X_train[index * batchSize::],
                                            y=[p[index * batchSize::], X_train[index * batchSize::]])
                index, flag = 0,1
                y_pred = q.argmax(1)
                ''' Accuracy is shown only for visualization, but it is not taken into account, as this is an unsupervised framework '''
                acc = np.round(clustering_accuracy(y_train, y_pred), 4)
                print('[INFO] Epoch ' + str(e) + ':  ----------->  L: ', np.round(loss[0], 6), \
                      '  Lr: ', np.round(loss[1], 6), '  Lc: ', np.round(loss[2], 6), '  Acc: ', acc)
                #print('[INFO] Epoch ' + str(e) + ':  ----------->  loss: ', np.round(loss, 4), '  Acc: ', acc)
            else:
                loss = model.train_on_batch(x=X_train[index * batchSize:(index + 1) * batchSize],
                                            y=[p[index * batchSize:(index + 1) * batchSize],
                                               X_train[index * batchSize:(index + 1) * batchSize]])
                index += 1
        flag = 0

        if early_stopping:
            # Early Stopping
            if params_DC['hybrid']:
                loss = loss[2]
            best, patience, save_DC = call_early_stopping(acc, loss, best, 'loss', patience)
            if save_DC:
                save_model(model, folders['models'] + 'DC/' + name_to_save + '.hdf5')
            if patience>patience_max:
                break
        else:
            save_model(model, folders['models'] + 'DC/' + name_to_save + '.hdf5')
    return model

# target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# Early stopping for DC model
def call_early_stopping(acc, loss, best, flag_ES='loss', patience=0):
    # Early Stopping
    if flag_ES == 'loss':
        new_value = loss
        decision = new_value<best
    elif flag_ES == 'acc': # This is supervised
        new_value = acc
        decision = new_value>best

    if decision: # if the new value of loss or accuracy is better than the best
        print(colored('[INFO] ' + flag_ES + ' improved from ' + str(best) + ' to ' + str(new_value) + ' ... saving model', 'red'))
        best = new_value
        patience = 0
        save_DC = True
    else:
        save_DC = False
        patience += 1

    return best, patience, save_DC

# Clustering accuracy
def clustering_accuracy(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    #print(y_true)
    #print(y_pred)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

# Extract clustering predictions
def extract_clustering_predictions(y_true, y_pred, num_classes):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)

    y_p = np.copy(y_pred)
    for i in range(num_classes):
        y_p[y_pred==i] = ind[i,1]
    return y_p
