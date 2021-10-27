# When problems with CUDA libraries
# sudo ldconfig /usr/local/cuda-9.0/lib64/

import os
# USE THE SELECTED GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.utils import to_categorical
import time
from tabulate import tabulate
import numpy as np
import pandas as pd
from Functions import load_data, fit_CAE, fit_DC, clustering_accuracy, extract_clustering_predictions
from Architectures import build_CAE
import warnings
from tensorflow.keras.models import load_model, save_model, Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
import random
from tensorflow.keras.layers import Layer, InputSpec
from sklearn.cluster import KMeans, SpectralClustering
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dropout, GlobalMaxPool2D, GlobalAvgPool2D, Flatten, concatenate, Multiply
from utils import plot_learning_curves, extract_TSNE, report_classification_results, plot_confusion_matrix, CAMs_computation


# GETTING REPRODUCIBLE THE RESULTS WITH KERAS
from numpy.random import seed
seed(1)
tf.random.set_seed(2)
#--------------------------------------------
warnings.filterwarnings('ignore')

# ------------------------------------------------------------- TRAIN ----------------------------------------------------------------------- #

def Train_CAE(folders, data_opt, params_DC, exp, flag_eval):
    t = time.time()

    # LOAD DATA
    print('---------\n[INFO] Loading Training data...')
    csv_train = pd.read_csv(folders['excels'] + 'GLOBAL.csv', sep=';')
    X_train, _ = load_data(folders, data_opt, csv_train)
    print('Time to load data: ', time.time() - t)

    # LOAD ARCHITECTURE
    print('---------\n[INFO] Loading architecture...')
    shape_images = {'rows': data_opt['x'], 'cols': data_opt['y'], 'depth': data_opt['z']}
    _, model = build_CAE(shape_images, params_DC)

    # TRAIN MODEL
    print('---------\n[INFO] Training model...')
    H, CAE, name_to_save = fit_CAE(folders, X_train, params_DC, model, exp)

    # PLOT LEARNING CURVES
    plot_learning_curves(H, folders['learn_curves'], 'CAE_' + name_to_save, partition='val', curve='loss')

    if flag_eval == True:
        if not os.path.exists(folders['general'] + 'reconstructed_images/exp_' + str(exp)):
            tf.io.gfile.mkdir(folders['general'] + 'reconstructed_images/exp_' + str(exp))
        for i in range(10):
            id_rand = random.choice(range(np.shape(X_train)[0]))
            x = np.expand_dims(X_train[id_rand], 0)
            predicted_image = CAE.predict(x)
            comp = np.concatenate((X_train[id_rand,:,:,:], predicted_image[0,:,:,:]),1)
            save_img(folders['general'] + 'reconstructed_images/exp_' + str(exp) + '/' + csv_train.ID[id_rand], comp)
    return CAE


def Train_DC_Model(folders, CAE_name, data_opt, params_DC, exp):

    CAE = load_model(folders['models'] + 'CAE/' + CAE_name)

    # LOAD DATA
    print('---------\n[INFO] Loading data...')
    csv_train = pd.read_csv(folders['excels'] + 'GLOBAL.csv', sep=';')
    X_train, y_train = load_data(folders, data_opt, csv_train)

    # Extract latent space from the CAE
    output_encoder = CAE.get_layer('bottle_neck').output
    output_encoder = GlobalMaxPool2D()(output_encoder)
    encoder = Model(inputs=CAE.input, outputs=output_encoder)

    # Define Deep Clustering Architecture
    n_clusters = params_DC['n_clusters']
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(output_encoder)
    if params_DC['hybrid'] == True:
        model = Model(inputs=CAE.input, outputs=[clustering_layer, CAE.output])
    else:
        model = Model(inputs=CAE.input, outputs=clustering_layer)
    print(model.summary())

    # Initialise clustering weigths
    latent_space = KMeans(n_clusters=n_clusters, n_init=20)
    # latent_space = SpectralClustering(n_clusters=3, n_init=20, random_state=42)
    _ = latent_space.fit_predict(encoder.predict(X_train))
    model.get_layer(name='clustering').set_weights([latent_space.cluster_centers_])

    # TRAIN MODEL
    print('---------\n[INFO] Training model...')
    fit_DC(folders, X_train, y_train, params_DC, model, exp)

    print('MODEL TRAINED')

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

# ------------------------------------------------------------- EVALUATION ----------------------------------------------------------------------- #
def Evaluate_DC_model(folders, model_name, data_opt, classes, class_mean, flag_mean):
    s, e, ppv, npv, f1, acc, auc = [], [], [], [], [], [], []
    S, E, PPV, NPV, F1, ACC, AUC = ['Sensitivity'], ['Specificity'], ['PPV (precision)'], ['NPV'], ['F1-score'], ['Accuracy'], ['AUC']

    folder_csv = pd.read_csv(folders['excels'] + 'GLOBAL.csv', sep=';')

    # Load data
    print('---------\n[INFO] Loading data...')
    X_val, y_true = load_data(folders, data_opt, folder_csv)

    # Load model
    model = load_model(folders['models'] + 'DC/' + model_name, custom_objects={'ClusteringLayer': ClusteringLayer})

    # Load y_pred
    preds = model.predict(X_val)
    if len(preds)==2:
        preds=preds[0]
    y_pred = np.argmax(preds, axis=1)
    y_pred = extract_clustering_predictions(y_true, y_pred, 3)

    '''
    if partition_dic['conf_mat'] == True:
        dirout = folders['general'] + 'confusionMatrix/' + model_name[0:-4] + 'png'
        plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues, dir_out=dirout)

    if partition_dic['CAMs'] == True:
        ## CAMs
        dim_rows, dim_cols = data_opt['x'], data_opt['y']
        layerToPredict = model.layers[32]
        folderToSave = folders['general'] + 'CAMs'
        CAMs_computation(folders, dim_rows, dim_cols, q, model, layerToPredict, folderToSave, folder_csv)
    '''
    # Show the average of results
    if flag_mean == True:
        res = report_classification_results(y_true, y_pred, classes, flag_mean, class_mean + 1)
        s.append(float(res['S'])), e.append(float(res['E'])), ppv.append(float(res['PPV'])), npv.append(
            float(res['NPV']))
        f1.append(float(res['F1'])), acc.append(float(res['ACC'])), auc.append(float(res['AUC']))
    else:
        report_classification_results(y_true, y_pred, classes, flag_mean, class_mean + 1)

    if flag_mean == True:
        headers = ['AVG', 'STD']

        S.append(np.round(np.mean(s), 4)), S.append(np.round(np.std(s), 4))
        E.append(np.round(np.mean(e), 4)), E.append(np.round(np.std(e), 4))
        PPV.append(np.round(np.mean(ppv), 4)), PPV.append(np.round(np.std(ppv), 4))
        NPV.append(np.round(np.mean(npv), 4)), NPV.append(np.round(np.std(npv), 4))
        F1.append(np.round(np.mean(f1), 4)), F1.append(np.round(np.std(f1), 4))
        ACC.append(np.round(np.mean(acc), 4)), ACC.append(np.round(np.std(acc), 4))
        AUC.append(np.round(np.mean(auc), 4)), AUC.append(np.round(np.std(auc), 4))

        my_data = [tuple(S), tuple(E), tuple(PPV), tuple(NPV), tuple(F1), tuple(ACC), tuple(AUC)]

        print(tabulate(my_data, headers=headers))


