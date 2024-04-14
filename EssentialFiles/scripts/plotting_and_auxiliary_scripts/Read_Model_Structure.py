import os
import numpy as np
import pickle
import json
import csv
import matplotlib.pyplot as plt
import matplotlib as mplt
from sklearn.metrics import roc_curve, log_loss
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import seaborn as sns
from tensorflow import keras
import tensorflow as tf
from numpy.random import seed
from datetime import datetime
from sklearn.metrics import confusion_matrix
from keras import activations
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Concatenate, Input
from keras import regularizers
from sklearn.utils import shuffle
from tensorflow import expand_dims

from sklearn.metrics import r2_score


def print_model_history(DIR):
    model = keras.models.load_model(f'{DIR}/KerasModel')
    print(model.summary())
    return


def main():
    MODEL_A_DIR = '/users/qf20170/project_folder/alt_model/ModelA/model_a_layer_size_test/20240227--203709'
    SIMP_REG_CLASS = '/users/qf20170/project_folder/regression_classification_test_models/weighted_reg_clas-model_retest--relu/20240212--234314'
    MODEL_B_DIR = '/users/qf20170/project_folder/alt_model/ModelB/model_b_layer_size_test/20240228--223246'
    REG_ONLY = '/users/qf20170/project_folder/regression_only/reg_only_model_retest/20240409--184614'
    CLAS_ONLY = '/users/qf20170/project_folder/classification_only/base-model/20240308--151522'
    
    print('clas_only')
    print_model_history(CLAS_ONLY)
    print('reg_only')
    print_model_history(REG_ONLY)
    print('simp_regclas')
    print_model_history(SIMP_REG_CLASS)
    print('model_a')
    print_model_history(MODEL_A_DIR)
    print('model_b')
    print_model_history(MODEL_B_DIR)
    
    return



if __name__ == "__main__":
    main()