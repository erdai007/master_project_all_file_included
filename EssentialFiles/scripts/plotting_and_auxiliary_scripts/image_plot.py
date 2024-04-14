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

plt.rcParams.update({'font.size':16})

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
    # idx = expand_dims(idx, -1)
    # x = expand_dims(x, -1)#Add a line for processed x
    # y = expand_dims(y, -1)
    # additional = expand_dims(additional, -1)#Add a line for processed ht

    return {'idx': idx, 'x': x, 'ht': additional, 'y': y}


def main():
    TESTDATA_DIR = '/storage/2/ek19824/ML_data/nosat_dataset/dataset_testing.pickle'
    test_data=prepare_dataset(TESTDATA_DIR)
    
    images=test_data['x']
    y = test_data['y']
    
    
    image_signal =  images[y==1.0]
    image_background = images[y==0.0]
    
    #print(image_signal.shape)
    #print('hello world')
    date = datetime.now().strftime("%Y%m%d--%H%M%S")
    
    #print(np.max(image_signal[0]))

    fig_signal, ax_signal=plt.subplots(1,1)
    fig_background, ax_background=plt.subplots(1,1)
    
    imp_signal =ax_signal.imshow(image_signal[0],origin='lower')
    ax_signal.set_axis_off()
    fig_signal.colorbar(imp_signal,ax=ax_signal)
    fig_signal.suptitle('signal')
    fig_signal.tight_layout()
    
    #imp_background = ax_background.imshow(image_background[0],origin='lower',vmax=np.max(image_signal[0]))
    imp_background = ax_background.imshow(image_background[0],origin='lower')

    ax_background.set_axis_off()
    fig_background.colorbar(imp_background,ax=ax_background)
    fig_background.suptitle('background')    
    fig_background.tight_layout()

    
    fig_signal.savefig(f'plots/images/{date}-signal_image.jpg')
    fig_background.savefig(f'plots/images/{date}-background_image.jpg')
    
    
    
    return



if __name__ == "__main__":
    main()
