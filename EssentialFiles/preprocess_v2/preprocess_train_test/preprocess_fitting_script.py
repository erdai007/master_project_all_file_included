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

import sklearn.preprocessing as preprocessing

import scipy.stats as stats
from scipy.special import inv_boxcox

def yeo_johnson_fit(x,lmda=None,scale=None):
    
    shape_x=x.shape
    x_buff=x.flatten()
    lmda_fit=None
    scale_fit=None

    if scale != None:
        scale_fit=float(scale)
    else:
        scale_fit=x_buff.max()
    x_buff = x_buff/scale_fit
    
    
    if lmda != None:
        lmda_fit=float(lmda)
        x_buff = stats.yeojohnson(x_buff,lmbda=lmda_fit)[0]
    else:
        x_buff, lmda_fit = stats.yeojohnson(x_buff)    
    x_return= x_buff.reshape(shape_x)
    
    return x_return, scale_fit, lmda_fit


def box_cox_fit(x,lmda=None,scale=None,shift=1e-5):
    
    shape_x=x.shape
    x_buff=x.flatten()
    lmda_fit=None
    scale_fit=None

    if scale != None:
        scale_fit=float(scale)
    else:
        scale_fit=x_buff.max()
    x_buff = x_buff/scale_fit + shift
    
    
    if lmda != None:
        lmda_fit=float(lmda)
        x_buff = stats.boxcox(x_buff,lmbda=lmda_fit)[0]
    else:
        x_buff, lmda_fit = stats.boxcox(x_buff)    
    x_return= x_buff.reshape(shape_x)
    
    return x_return, scale_fit, lmda_fit

def inv_yeojohnson(x,lmda):
    y=0.0
    if (x>=0 and lmda != 0):
        y = (lmda*x+1)**(1/lmda) - 1
    elif (x>=0 and lmda == 0):
        y = np.exp(x) - 1
    elif (x<0 and lmda != 2):
        y = 1 - ( -(2-lmda)*x +1 )**(1/(2-lmda))
    elif (x<0 and lmda == 2):
        y = 1 - np.exp(-x)
    
    return y

def inv_yeo_johnson_trans(x,lmda,scale):
    shape_x = x.shape
    x_buff = x.flatten()
    trans = np.vectorize(inv_yeojohnson,excluded=['lmda'])
    x_buff = trans(x_buff,lmda)
    x_buff = x_buff * scale
    x_return = x_buff.reshape(shape_x)
    return x_return

def inv_box_cox_trans(x,lmda,scale, shift=1e-5):
    shape_x = x.shape
    x_buff = x.flatten()
    trans = np.vectorize(inv_boxcox)
    x_buff = trans(x_buff,lmbda=lmda)
    x_buff = (x_buff - shift) * scale
    x_return = x_buff.reshape(shape_x)    
    return x_return




def prepare_dataset(dataset):

    f=open(dataset, 'rb')
    data = pickle.load(f) 
    idx = data['idx'] 
    x = data["images"]
    y = data["labels"]
    additional = data[b'phaseIPFJetHt']
    sf = 4550            #4550
    additional = additional/sf
    idx, x, additional, y = shuffle(idx, x, additional, y)      # need shuffle to ensure the validation set is not all background (not done by shuffle arg in model.fit())
    idx = expand_dims(idx, -1)
    x = expand_dims(x, -1)#Add a line for processed x
    y = expand_dims(y, -1)
    additional = expand_dims(additional, -1)#Add a line for processed ht

    return {'idx': idx, 'x': x, 'ht': additional, 'y': y}



def transform_train(x,save=True,filename='Name'):
    
    #yeo-johnson
    x_ys, scale_ys, lmda_ys = yeo_johnson_fit(x)
    #box-cox'
    x_bc, scale_bc, lmda_bc = box_cox_fit(x)
    stats = [{'entry':'yeojohnson scale factor','val':scale_ys},
             {'entry':'yeojohnson lambda','val':lmda_ys},
             {'entry':'boxcox scale factor','val':scale_bc},
             {'entry':'boxcox lambda','val':lmda_bc}]
    date = datetime.now().strftime("%Y%m%d--%H%M%S")
    
    if save:
        with open(f'stats/{date}_{filename}.csv','w') as csvfile:
            writer= csv.DictWriter(csvfile, fieldnames=['entry','val'])
            writer.writeheader()
            writer.writerows(stats)
        print('Training parameters saved.')
    
    
    return {'ys':[x_ys,scale_ys,lmda_ys],'bc':[x_bc, scale_bc, lmda_bc]}


def transform_hist(x, x_trans,title='hist_title'):
    
    fig, (ax_original, ax_preprocessed) = plt.subplots(1,2,sharey=False,sharex=False)
    ax_original.hist(x,bins=100,range=[0.0,x.max()],density=True,color='blue')
    ax_preprocessed.hist(x_trans,bins=100,range=[0.0,x_trans.max()],density=True,color='orange')
    ax_original.set_title('Original pixel distribution')
    ax_preprocessed.set_title('Pixel distribution after preprocess')
    fig.suptitle(title)
    fig.set_size_inches(14.5,7.0)
    fig.set_dpi(200)
    date = datetime.now().strftime("%Y%m%d--%H%M%S")
    fig.savefig(f'figs/{date}_{title}.jpg')
    print('Plot completed')
    return




def main():
    TRAINDATA_DIR = '/storage/2/ek19824/ML_data/nosat_dataset/dataset_training.pickle'
    train_data = prepare_dataset(TRAINDATA_DIR)
    x_data_raw=train_data['x'].numpy()
    ht_data_raw=train_data['ht'].numpy()
    
    x_data=x_data_raw.reshape(*x_data_raw.shape[:-2],-1)
    ht_data=ht_data_raw.reshape(-1)
    
    print(x_data.shape)
    print(ht_data.shape)
    
    print(x_data.flatten().max())#1752.25
    print(ht_data.flatten().max())#0.9998818358727788
    
    print(x_data.flatten().min())#0.0
    print(ht_data.flatten().min())#0.0
    
    x_train=x_data.flatten()
    #x_train=x_train[x_train>0]
    
    ht_train=ht_data.flatten()
    #ht_train=ht_train[ht_train>0]
    
    print(x_train.shape)#6284969
    print(ht_train.shape)#90178
    
    print('Training begins')
    
    x_transform = transform_train(x_train,filename='x_transform')
    ht_transform = transform_train(ht_train,filename='ht_transform')
    
    print('Training completed, plotting data distribution')
    
    transform_hist(x_train[x_train>0],x_transform['ys'][0],title='x_yeo-johnson')
    transform_hist(x_train[x_train>0],x_transform['bc'][0],title='x_box-cox')
    transform_hist(ht_train[ht_train>0],ht_transform['ys'][0],title='ht_yeo-johnson')
    transform_hist(ht_train[ht_train>0],ht_transform['bc'][0],title='ht_box-cox')    
    
    print('Plotting completed')
    
    
    
    return

if __name__ == "__main__":
    main()