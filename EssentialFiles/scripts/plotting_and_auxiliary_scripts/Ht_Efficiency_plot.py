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
from numpy.random import shuffle as nshuffle

from sklearn.metrics import r2_score

def prepare_dataset(dataset):

    f=open(dataset, 'rb')
    data = pickle.load(f) 
    idx = data['idx'] 
    x = data["images"]
    y = data["labels"]
    additional = data[b'phaseIPFJetHt']
    ht_gen = data[b'genJetHt']
    sf = 4550            #4550
    additional = additional/sf
    ht_gen = ht_gen/sf
    # idx, x, additional, y = shuffle(idx, x, additional, y)      # need shuffle to ensure the validation set is not all background (not done by shuffle arg in model.fit())
    # idx = expand_dims(idx, -1)
    # x = expand_dims(x, -1)#Add a line for processed x
    # y = expand_dims(y, -1)
    # additional = expand_dims(additional, -1)#Add a line for processed ht

    return {'idx': idx, 'x': x, 'ht': additional, 'ht_gen':ht_gen,'y': y}

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

def rule_based(y,ht,htgen,ht_cut):
    criterion = (ht<=ht_cut)#criterion for discarding a signal
    y_pred=y
    y_pred[criterion]=0.0
    output_data_set = {'y':y_pred,'ht':htgen,'min_thr':0.50}
    return output_data_set


def Ht_Eff_plot(y_pred,thr,y_true,ht,sample_num=50):
    Ht_range, Ht_diff= np.linspace(0.0,1.0,sample_num+1,retstep=True)
    Ht_upper = Ht_range[1:]
    Ht_lower = Ht_range[:-1]
    Ht_interval = np.transpose(np.vstack((Ht_lower,Ht_upper)))
    Ht_midpoint = (Ht_upper + Ht_lower)/2
    ht_flatten= ht.flatten()
    
    
    y_thr= y_pred-(thr-0.5)
    y_thr[y_thr>=1.5]=1.499999
    y_thr[y_thr<=-0.5]=-0.5
    sig_pred= y_thr.round().flatten()
    sig_true= np.array(list(map(int,list(y_true)))).flatten()
    
    efficiency = []
    
    for i in Ht_interval.tolist():
        lower_bound = i[0]
        upper_bound=i[1]
        
        
        signal_num= sig_true[(ht_flatten<=upper_bound)&(ht_flatten>=lower_bound)&(sig_true==1)].size
        tp_num= sig_pred[(ht_flatten<=upper_bound)&(ht_flatten>=lower_bound)&(sig_true==1)&(sig_pred==1)].size
        
        if signal_num!=0:
            efficiency.append(tp_num/signal_num)
        elif len(efficiency)==0:
            efficiency.append(0.0)
        else:
            efficiency.append(efficiency[-1])
        
    Ht_plot = Ht_midpoint * 4550.0
    
    return {'ht':Ht_plot,'eff':np.array(efficiency),'width':Ht_diff/2*4550.0}


def main():
    TESTDATA_DIR = '/storage/2/ek19824/ML_data/nosat_dataset/dataset_testing.pickle'
    MODEL_A_PRED_DIR = '/users/qf20170/project_folder/alt_model/ModelA/model_a_layer_size_test/20240227--203709/predictions.pickle'
    SIMPLE_REG_CLAS_PRED_DIR = '/users/qf20170/project_folder/regression_classification_test_models/weighted_reg_clas-model_retest--relu/20240212--234314/predictions.pickle'
    
    test_data = prepare_dataset(TESTDATA_DIR)
    model_a_data= read_off_predicition(MODEL_A_PRED_DIR)
    simple_regclas_data = read_off_predicition(SIMPLE_REG_CLAS_PRED_DIR)
    rule_based_data = rule_based(test_data['y'],test_data['ht'],test_data['ht_gen'],370.0/4550.0)
    
    model_a_thr=ROC_data(model_a_data['y'],test_data['y'])['thr']
    simple_regclas_thr=ROC_data(simple_regclas_data['y'],test_data['y'])['thr']
    
    
    ht_eff_rc_plot = Ht_Eff_plot(simple_regclas_data['y'],simple_regclas_thr,test_data['y'],simple_regclas_data['ht'])
    ht_eff_a = Ht_Eff_plot(model_a_data['y'],model_a_thr,test_data['y'],model_a_data['ht'])
    ht_eff_rulebased = Ht_Eff_plot(rule_based_data['y'],rule_based_data['min_thr'],test_data['y'],test_data['ht_gen'])
    
    date = datetime.now().strftime("%Y%m%d--%H%M%S")
    plt.figure(dpi=200)
    plt.plot(ht_eff_rulebased['ht'],ht_eff_rulebased['eff'],label='Rule-Based',linestyle='-',linewidth=3.0)
    plt.plot(ht_eff_a['ht'],ht_eff_a['eff'],label='Model A',linestyle='-',linewidth=3.0)
    plt.plot(ht_eff_rc_plot['ht'],ht_eff_rc_plot['eff'],label='Simple Model',linestyle='-',linewidth=3.0)
    plt.semilogx()
    plt.legend()
    plt.ylim(-0.01,1.00)
    
    plt.xlabel(r'$H_T / $GeV')
    plt.ylabel('Signal Efficiency')
    plt.grid('True')
    plt.tight_layout()
    plt.savefig(f'plots/HtEff/{date}-Ht_Efficiency.png')
    
    return

if __name__ == "__main__":
    main()
