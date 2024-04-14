import os
import numpy as np
import pickle
import json
import csv
import matplotlib.pyplot as plt
import matplotlib as mplt
from sklearn.metrics import roc_curve, log_loss
from sklearn.metrics import confusion_matrix
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

#plt.rc('text',usetex=True)
plt.rcParams.update({'font.size':12})

def prepare_dataset(dataset):

    f=open(dataset, 'rb')
    data = pickle.load(f) 
    idx = data['idx'] 
    x = data["images"]
    y = data["labels"]
    additional = data[b'phaseIPFJetHt']
    sf = 4550            #4550
    additional = additional/sf
    # idx, x, additional, y = shuffle(idx, x, additional, y)      # need shuffle to ensure the validation set is not all background (not done by shuffle arg in model.fit())
    # idx = expand_dims(idx, -1)
    # x = expand_dims(x, -1)#Add a line for processed x
    # y = expand_dims(y, -1)
    # additional = expand_dims(additional, -1)#Add a line for processed ht

    return {'idx': idx, 'x': x, 'ht': additional, 'y': y}

def read_off_predicition(dataset,include_ht=True):
    f=open(dataset,'rb')
    data = pickle.load(f)
    idx = data['idx']
    #y_pred = data['score']
    ht_pred = data['ht']
    # if include_ht:
    #     ht_pred = data['ht']
    #     output_data_set = {'idx':idx, 'y':y_pred, 'ht':ht_pred}
    # else:
    #     output_data_set = {'idx':idx, 'y':y_pred}
    output_data_set = {'id':idx, 'ht':ht_pred}
    
    return output_data_set

# def ROC_data(y_pred,y_true):
#     fpr, tpr, thresholds = roc_curve(list(map(int,list(y_true))),y_pred)
#     fpr *= (40*10**3) * (2760 / 3564)
#     min_thr = thresholds[fpr<10][-1]
    
#     return {'tpr':tpr,'fpr':fpr,'thr':min_thr}



def main():
    TESTDATA_DIR = '/storage/2/ek19824/ML_data/nosat_dataset/dataset_testing.pickle'
    #CLAS_ONLY_MODEL_PRED_DIR = '/users/qf20170/project_folder/classification_only/base-model/20240308--151522/predictions.pickle'
    MODEL_A_PRED_DIR = '/users/qf20170/project_folder/alt_model/ModelA/model_a_layer_size_test/20240227--203709/predictions.pickle'
    REG_ONLY_PRED_DIR = '/users/qf20170/project_folder/regression_only/reg_only_model_retest/20240409--184614/predictions.pickle'
    REG_ONLY_WPRED_DIR = '/users/qf20170/project_folder/regression_only/reg_only_model_retest/20240409--182034/predictions.pickle'#Worst reg only prediction set
    MODEL_A_PRED_DIR = '/users/qf20170/project_folder/alt_model/ModelA/model_a_layer_size_test/20240227--203709/predictions.pickle'
    MODEL_B_PRED_DIR ='/users/qf20170/project_folder/alt_model/ModelB/model_b_layer_size_test/20240228--223246/predictions.pickle'
    SIMPLE_REG_CLAS_0p7_DIR ='/users/qf20170/project_folder/regression_classification_test_models/weighted_reg_clas-model_retest--relu/20240212--234314/predictions.pickle'
    SIMPLE_REG_CLAS_0p8_DIR = '/users/qf20170/project_folder/regression_classification_test_models/weighted_reg_clas-model_retest--relu/20240212--172727/predictions.pickle'
    
    SIMPLE_REG_CLAS_0p5_RELU_DIR = '/users/qf20170/project_folder/regression_classification_test_models/weighted_reg_clas-model_retest--relu/20240218--153235/predictions.pickle'
    SIMPLE_REG_CLAS_0p5_LNR_DIR = '/users/qf20170/project_folder/regression_classification_test_models/weighted_reg_clas-model_retest--linear/20240218--144745/predictions.pickle'
    SIMPLE_REG_CLAS_0p5_SFT_DIR = '/users/qf20170/project_folder/regression_classification_test_models/simple_reg_clas-model_retest/20240208--221104-softplus/predictions.pickle'
    
    # test_data=prepare_dataset(TESTDATA_DIR)
    # predicted_data = read_off_predicition(MODEL_PRED_DIR)
    # print(test_data['idx'].size)
    # print(predicted_data['idx'].size)
    # print(np.array_equal(test_data['idx'],predicted_data['idx']))
    test_data=prepare_dataset(TESTDATA_DIR)
    #clas_only_data=read_off_predicition(CLAS_ONLY_MODEL_PRED_DIR,include_ht=False)
    model_a_data=read_off_predicition(MODEL_A_PRED_DIR)
    model_b_data=read_off_predicition(MODEL_B_PRED_DIR)
    reg_only_data=read_off_predicition(REG_ONLY_PRED_DIR)
    reg_only_w_data=read_off_predicition(REG_ONLY_WPRED_DIR)
    simp_reg_clas_0p7_data=read_off_predicition(SIMPLE_REG_CLAS_0p7_DIR)
    simp_reg_clas_0p8_data=read_off_predicition(SIMPLE_REG_CLAS_0p8_DIR)
    relu_regclas_data=read_off_predicition(SIMPLE_REG_CLAS_0p5_RELU_DIR)
    lnr_regclas_data=read_off_predicition(SIMPLE_REG_CLAS_0p5_LNR_DIR)
    sft_regclas_data=read_off_predicition(SIMPLE_REG_CLAS_0p5_SFT_DIR)
    
    
    ht_test=test_data['ht']*4550.0
    
    #ht_pred=model_a_data['ht']*4550.0
    #ht_pred=model_b_data['ht']*4550.0
    #ht_pred=reg_only_data['ht']*4550.0
    #ht_pred=simp_reg_clas_0p7_data['ht']*4550.0
    #ht_pred=simp_reg_clas_0p8_data['ht']*4550.0
    #ht_pred=reg_only_data['ht']*4550.0
    #ht_pred=reg_only_w_data['ht']*4550.0
    
    #ht_pred=relu_regclas_data['ht']*4550.0
    #ht_pred=lnr_regclas_data['ht']*4550.0
    ht_pred=sft_regclas_data['ht']*4550.0
    
    
    
    
    reg_r2=r2_score(ht_test,ht_pred)
    
    plt.figure(dpi=200)
    plt.scatter(ht_test,ht_pred,c='blue',alpha=0.5,s=(mplt.rcParams['lines.markersize']*0.6)**2)
    plt.plot([0.0,4550.0],[0.0,4550.0],label='Ideal prediction',linestyle='-',color='red',linewidth=1.0)
    plt.xlabel(r'True $H_T$/GeV')
    plt.ylabel(r'Predicted $H_T$/GeV')
    plt.xlim(-0.03*4550,4550.0)
    plt.ylim(-0.03*4550,4550.0)
    plt.legend()
    plt.grid(True)
    r2_string= r'$R^2=$'+f'{reg_r2:.6f}'
    
    #plt.text(0.6*4550.0,0.05*4550.0,r2_string)
    #plt.title('Regression Performance')
    #plt.text(2000.0,4050.0,'Regression Performance',fontsize='xx-large')
    plt.tight_layout()
    
    date = datetime.now().strftime("%Y%m%d--%H%M%S")
    
    plt.savefig(f'plots/RegPlot/{date}-RegPlot.png')
    print(f'R^2= {reg_r2:.6f}')
    print(f'Plot saved at plots/RegPlot/{date}-RegPlot.png')

    
    
    
    return

if __name__ == "__main__":
    main()
