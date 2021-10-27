
import os
# USE THE SELECTED GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.optimizers import SGD, Adadelta
import warnings
from Methodology import Train_CAE, Train_DC_Model, Evaluate_DC_model
import tensorflow as tf


# GETTING REPRODUCIBLE THE RESULTS WITH KERAS
from numpy.random import seed
seed(1)
tf.random.set_seed(2)
#--------------------------------------------
warnings.filterwarnings('ignore')


# Important folders
folder = 'V:/Proyectos/BLADDER/CIBM/' # General path
#folder = '../data/' # General path
folders = {'general': folder,
           'excels': folder + 'excels/',
           'images': folder + 'images/',
           'models': folder + 'models/',
           'learn_curves': folder + 'learning_curves/'}


# Data options
data_options = {'x': 128, # rows
                'y': 128, # cols
                'z': 3, # depth
                'colorMode': 'rgb', # 'rgb' or 'greyscale'
                'classes': 3,
                'classMode': 'categorical'}

# Hyperparameters for deep clustering
params_DC = {
             'num_convs': 3,
             'first_filter': 64, # The number of filters is increasing by duplicating x2
             'feat_mode': {'option': 'ATT', # GMP, GAP, Flatten, Conv_1_1, GMP_Z, GAP_Z, ATT or None
                           'temp': 4}, # Only if option is 'Conv_1_1'. This is a temperature parameter to reduce the number of filters in the latent space (filters/temp)
             'reconstruction_type': 'Conv2DTranspose', # UpSampling or Conv2DTranspose
             'hyperparameters': {'numEpochs': 200, # maximum number of epochs
                             'lr': 0.5, # Learning rate
                             'lf_CAE': 'mse', # Loss function for reconstruction
                             'lf_DC': 'kld', # Loss function for deep clustering
                             'opti': Adadelta, # Optimizer
                             'bs': 32}, # Batch size
             'weights': [0.3, 1], # [0] classification weights, [1] reconstruction weights
             'n_clusters': 3,
             'hybrid': True # if hybrid=True, reconstruction and clustering branches are considered --> DCEAC, otherwise --> rDCEC
            }

# Number of GPUs
G = 1

# Number of experiment to train
exp = 1

if __name__ == '__main__':

    DC_operation = {'train_CAE': False,
                   'train_DC_model': True,
                   'eval_DC_model': False}


    # ------------------------------ DEEP CLUSTERING ------------------------------ #
    # Train the CAE to set the initial weights with the kmeans running on embedding features
    if DC_operation['train_CAE']:
        ''' if flag_eval==True, 10 random reconstructed images are stored to analyse the model's performance'''
        Train_CAE(folders, data_options, params_DC, exp, flag_eval=True)

    # Train the Deep Clustering model
    if DC_operation['train_DC_model']:
        autoencoder_name = 'Model_20_bs32_lr0.5_mse_Adadelta_3_64.hdf5'
        Train_DC_Model(folders, autoencoder_name, data_options, params_DC, exp)

    # Evaluate the Deep Clustering model
    if DC_operation['eval_DC_model']:
        classes = {'0': 'Non_tumour', '1': 'Low_grade', '2':'High_grade'}
        class_mean = 4  # class to obtain predictions -- When 3 classes, if class_mean==4 --> average of micro-avg
        '''
        partition_dic = {'parti': 'test',  # 'val' or 'test'
                         'CAMs': False,  # Show and save CAMs
                         'conf_mat': False,  # Show and save confusion matrix
                         'TSNE': False}  # Save embedding space
        '''
        model_name = 'Model_16_bs32_lr0.5_Adadelta_3_64_hybrid_True.hdf5'
        Evaluate_DC_model(folders, model_name, data_options, classes, class_mean, flag_mean=True)
