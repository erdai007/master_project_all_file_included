import os
import numpy as np
import pickle
import json
import csv
import matplotlib.pyplot as plt
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

plt.rcParams.update({'font.size':10})

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
    y_pred = data['score']
    if include_ht:
        ht_pred = data['ht']
        output_data_set = {'idx':idx, 'y':y_pred, 'ht':ht_pred}
    else:
        output_data_set = {'idx':idx, 'y':y_pred}
    
    return output_data_set

def ROC_data(y_pred,y_true):
    fpr, tpr, thresholds = roc_curve(list(map(int,list(y_true))),y_pred)
    fpr *= (40*10**3) * (2760 / 3564)
    min_thr = thresholds[fpr<10][-1]
    
    return {'tpr':tpr,'fpr':fpr,'thr':min_thr}




def main():
    TESTDATA_DIR = '/storage/2/ek19824/ML_data/nosat_dataset/dataset_testing.pickle'
    CLAS_ONLY_MODEL_PRED_DIR = '/users/qf20170/project_folder/classification_only/base-model/20240308--151522/predictions.pickle'
    CLAS_ONLY_WORST_MODEL_PRED_DIR='/users/qf20170/project_folder/classification_only/base-model/20240308--150430/predictions.pickle'
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
    clas_only_data=read_off_predicition(CLAS_ONLY_MODEL_PRED_DIR,include_ht=False)
    clas_only_w_data=read_off_predicition(CLAS_ONLY_WORST_MODEL_PRED_DIR,include_ht=False)
    model_a_data=read_off_predicition(MODEL_A_PRED_DIR)
    model_b_data=read_off_predicition(MODEL_B_PRED_DIR)
    simple_regclas_0p7_data=read_off_predicition(SIMPLE_REG_CLAS_0p7_DIR)
    simple_regclas_0p8_data=read_off_predicition(SIMPLE_REG_CLAS_0p8_DIR)
    
    relu_regclas_data=read_off_predicition(SIMPLE_REG_CLAS_0p5_RELU_DIR)
    lnr_regclas_data=read_off_predicition(SIMPLE_REG_CLAS_0p5_LNR_DIR)
    sft_regclas_data=read_off_predicition(SIMPLE_REG_CLAS_0p5_SFT_DIR)
    
    ROC_clas_only = ROC_data(clas_only_data['y'],test_data['y'])
    ROC_clas_only_w = ROC_data(clas_only_w_data['y'],test_data['y'])
    ROC_model_a=ROC_data(model_a_data['y'],test_data['y'])
    ROC_model_b=ROC_data(model_b_data['y'],test_data['y'])
    ROC_simple_regclas_0p7=ROC_data(simple_regclas_0p7_data['y'],test_data['y'])
    ROC_simple_regclas_0p8=ROC_data(simple_regclas_0p8_data['y'],test_data['y'])
    
    ROC_relu_regclas= ROC_data(relu_regclas_data['y'],test_data['y'])
    ROC_lnr_regclas= ROC_data(lnr_regclas_data['y'],test_data['y'])
    ROC_sft_regclas= ROC_data(sft_regclas_data['y'],test_data['y'])
    
    
    # with open(f'data/ROC_data.pickle','wb') as f:
    #     pickle.dump(ROC_clas_only,f)
    #     f.close()
    
    date = datetime.now().strftime("%Y%m%d--%H%M%S")
    plt.figure(dpi=200)
    # plt.plot(ROC_clas_only['fpr'],ROC_clas_only['tpr'],label='Classification Only',linestyle='-',linewidth=3.0,color='blue')
    # plt.plot(ROC_model_a['fpr'],ROC_model_a['tpr'],label='New Model',linestyle='-',linewidth=3.0,color='orange')
    
    # plt.plot(ROC_clas_only['fpr'],ROC_clas_only['tpr'],label='Baseline Model(Best)',linestyle='--',linewidth=1.0)
    # plt.plot(ROC_clas_only_w['fpr'],ROC_clas_only_w['tpr'],label='Baseline Model(Worst)',linestyle='--',linewidth=1.0)
    # plt.plot(ROC_simple_regclas_0p7['fpr'],ROC_simple_regclas_0p7['tpr'],label='Simple Model',linestyle='-',linewidth=1.0)
    # plt.plot(ROC_simple_regclas_0p7['fpr'],ROC_simple_regclas_0p7['tpr'],label=r'Simple Model($\gamma=0.7 $)',linestyle='-',linewidth=1.0)
    # plt.plot(ROC_simple_regclas_0p8['fpr'],ROC_simple_regclas_0p8['tpr'],label=r'Simple Model($\gamma=0.8 $)',linestyle='-',linewidth=1.0)
    # plt.plot(ROC_model_a['fpr'],ROC_model_a['tpr'],label='Model A',linestyle='-',linewidth=1.0)
    # plt.plot(ROC_model_b['fpr'],ROC_model_b['tpr'],label='Model B',linestyle='-',linewidth=1.0)
    plt.plot(ROC_relu_regclas['fpr'],ROC_relu_regclas['tpr'],label='ReLU',linestyle='-',linewidth=1.0)
    plt.plot(ROC_lnr_regclas['fpr'],ROC_lnr_regclas['tpr'],label='Linear',linestyle='-',linewidth=1.0)
    plt.plot(ROC_sft_regclas['fpr'],ROC_sft_regclas['tpr'],label='SoftPlus',linestyle='-',linewidth=1.0)
    
    
    plt.plot([1.0e1,1.0e1],[0.0,1.0],linestyle=':',linewidth=1.3,color='red')
    plt.semilogx()
    
    plt.xlabel("Triggering Rate/kHz")
    plt.ylabel("Signal Efficiency")    
    # plt.xlim(0.1,40000)
    plt.xlim(5.0,30)
    plt.ylim(0.45,0.51)
    
    plt.grid(True)
    plt.legend()
    #plt.title('ROC')
    #plt.text(1.9e1,0.3,'ROC',fontsize='xx-large')
    plt.tight_layout()
    plt.savefig(f'plots/ROC/{date}-ROC.png')
    print(f'Plot saved at plots/ROC/{date}-ROC.png')
    
    
    
    return

if __name__ == "__main__":
    main()
