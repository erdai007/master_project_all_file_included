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


def create_keras_model(nBinsX, nBinsY):
        
    img_input = Input(shape=(nBinsX, nBinsY, 1))

    conv1 = Conv2D(1, (5, 5),kernel_regularizer=regularizers.l2(1e-5), padding='same', activation='relu', name='conv2d1')(img_input)
    conv1_bn = BatchNormalization(axis=-1)(conv1)
    pool1 = MaxPooling2D((4, 4), padding='valid', name='pool2d1')(conv1_bn)

    conv2 = Conv2D(1, (3, 3), padding='same', activation='relu', name='conv2d2')(pool1)
    conv2_bn = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2), padding='valid', name='pool2d2')(conv2_bn)

    flat = Flatten()(pool2)

    #additional_input = Input(shape=(1))

    #merged_layer = Concatenate()([flat, additional_input])

    dense1 = Dense(64, activation='relu')(flat)
    dense2 = Dense(32, activation='relu')(dense1)
    dense3 = Dense(8, activation='relu')(dense2)
    final_bn = BatchNormalization(axis=-1)(dense3)

    #classification_output = Dense(1, activation='sigmoid', name="classification_output")(final_bn)
    regression_output = Dense(1, activation='relu',name="regression_output")(final_bn)


    # Define the model with both the image and additional input as inputs
    model = tf.keras.models.Model(inputs=img_input, outputs=regression_output)
    
    
    regression_loss = 'mean_squared_error'
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss=regression_loss, optimizer=opt, metrics=regression_loss)
    
    return model

def data_preprocess():
    return


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


class Model:
   
    def __init__(self, outdir) -> None:
        self.outdir = outdir
        pass


    def new_model(self):
        self.model = create_keras_model(78, 78)
        self.outdir = f"{self.outdir}"
        now = datetime.now()
        self.modeldir = now.strftime("%Y%m%d--%H%M%S")
        self.model_name = "KerasModel"
        os.makedirs(f'{self.outdir}/{self.modeldir}', exist_ok=False)


    def load_model(self, modeldir):
        self.model = keras.models.load_model(f'{self.outdir}/{modeldir}/KerasModel')
        self.modeldir = modeldir
        self.model_name = 'KerasModel'


    def fit_model(self, train_data, save=True):
       
        print(self.model.summary())

        # Example for setting sample_weight in model.fit
        sample_weights = np.array(tf.reshape(train_data['y'],shape=[-1]))
        sample_weights[sample_weights==1] = 1.0
        sample_weights[sample_weights==0] = 1.0

        self.history = self.model.fit(train_data['x'], train_data['ht'],
                            epochs=40,
                            shuffle=True,      
                            verbose=2,
                            batch_size=250,
                            validation_split=0.25,
                            sample_weight=sample_weights,
                            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])#Possible locations for modification

        if save:
            self.model.save(f"{self.outdir}/{self.modeldir}/{self.model_name}")
            self.model.save_weights(f"{self.outdir}/{self.modeldir}/{self.model_name}.h5")

            metrics = dict(self.history.history)
            with open(f"{self.outdir}/{self.modeldir}/metrics_{self.model_name}.json", 'w') as f:
                json.dump(metrics , f)
            print(f"metrics saved: {self.outdir}/{self.modeldir}")


    def plot_training_curves(self, history=False, save=False):

        if not history:
            with open(f"{self.outdir}/metrics_{self.model_name}.json", 'r') as file:
                self.history = json.load(file)
                file.close()

        plt.figure(figsize=(10,5), dpi=200)
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=False)
        fig, (ax_reg_loss)=plt.subplots(1,1,sharey=False)

        # ax1.plot(self.history.history['accuracy'])
        # ax1.plot(self.history.history['val_accuracy'])
        # ax1.set_ylabel('accuracy')
        # ax1.set_xlabel('epoch')
        # ax1.legend(['train', 'validation'], loc='lower left')

        # ax2.plot(self.history.history['loss'])
        # ax2.plot(self.history.history['val_loss'])
        # ax2.set_ylabel('loss')
        # ax2.set_xlabel('epoch')
        # ax2.legend(['train', 'validation'], loc='upper left')

        # ax3.plot(self.history.history['sensitivity_at_specificity'])
        # ax3.plot(self.history.history['val_sensitivity_at_specificity'])
        # ax3.set_ylabel('SAS')
        # ax3.set_xlabel('epoch')
        # ax3.legend(['train', 'validation'], loc='upper left')
        
        ax_reg_loss.plot(self.history.history['mean_squared_error'])
        ax_reg_loss.plot(self.history.history['val_mean_squared_error'])
        ax_reg_loss.set_ylabel('regression error')
        ax_reg_loss.set_xlabel('epoch')
        ax_reg_loss.legend(['train','validation'],loc='lower right')

        plt.title(f"Training Curves",loc='center')  
        plt.tight_layout()
        if save:
          plt.savefig(f"{self.outdir}/{self.modeldir}/ntraining_curves.jpg")
          print('Training Curves Plotted')
        plt.show()


    def test_model(self, test_data, save=True, verbose=False):
        
        if verbose: 
          print(self.model.summary())

        self.scores = self.model.predict(test_data['x'])
        self.regress_val = self.model.predict(test_data['x'])
        idx = np.array(tf.squeeze(test_data['idx']))
        sorted_idx = np.argsort(idx)
        results = {'idx': idx[sorted_idx], 'ht':self.regress_val[sorted_idx]}

        if save:
          with open(f'{self.outdir}/{self.modeldir}/predictions.pickle', 'wb') as f:
            pickle.dump(results, f)
            f.close()

        print(f'Predictions saved to {self.outdir}/{self.modeldir}/predictions.pickle')

    def plot_regression_performance(self, ht, save=True):
        ht_pred=self.regress_val
        ht_true=ht
        self.reg_r2=r2_score(ht_true,ht_pred)
        
        reorder_hts=sorted(zip(ht_true,ht_pred), key=lambda pair: pair[0])
        
        plot_x=[x for x,_ in reorder_hts]
        plot_y=[y for _,y in reorder_hts]
        
        
        plt.figure(dpi=200)
        #plt.plot(plot_x,plot_y,label='CNN',linestyle='-',color='blue',linewidth=1.0)
        plt.scatter(plot_x,plot_y,c='blue',alpha=0.5,s=(mplt.rcParams['lines.markersize']*0.6)**2)
        plt.plot([0.0,0.8],[0.0,0.8],label='Ideal prediction',linestyle='-',color='red',linewidth=1.0)
        plt.xlabel('True Ht')
        plt.ylabel('Predicted Ht')
        
        plt.xlim(-0.03,0.9)
        plt.ylim(-0.03,0.9)
        
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.text(0.4,0.05,'R^2={:5f}'.format(self.reg_r2))
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.outdir}/{self.modeldir}/reg_comparison')
            print('Regression Performance plotted')
        


    def plot_ROC(self, y, save=True):
      fpr, tpr, thresholds = roc_curve(list(map(int, list(y))), self.scores)
      fpr *= (40*10**3) * (2760 / 3564)
      self.min_thr = thresholds[fpr<10][-1]
      self.eff = tpr[thresholds==self.min_thr][0]
      plt.figure(dpi=200) 
      plt.plot(fpr,tpr,label='CNN',linestyle='-', color='orange', linewidth=3.0)
      #plt.plot(tpr_ht,fpr_ht,label='QuadJetHT (no pT selection)',linestyle='-')
      plt.semilogx()
      plt.xlabel("Rate kHz")
      plt.ylabel("Signal Efficiency")
      plt.xlim(0.1,40000)
      plt.grid(True)
      plt.legend(loc='upper left')
      plt.tight_layout()
      if save:
        plt.savefig(f'{self.outdir}/{self.modeldir}/roc.png')
        print('ROC Plotted')


    def plot_confusion_matrix(self, y, save=False):
        
        scoresthr = np.array(self.scores - (self.min_thr - 0.5))
        matrix_confusion = confusion_matrix(list(map(int, list(y))), scoresthr.round())
        tn, fp, fn, tp = matrix_confusion.ravel()

        plt.figure(figsize=(10,8), dpi=200)
        sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)
        plt.ylabel("y true")
        plt.xlabel("y pred")
        plt.title(f"Threshold: {self.min_thr}",loc='center')
        if save:
          plt.savefig(f'{self.outdir}/{self.modeldir}/nconfusion_matrix.jpg')
          print('Confusion Matrix Plotted')

        self.acc = (tn + tp) / (tn + fp + fn + tp)
        self.pre = tp / (tp + fp)
        self.f1s = 2 / ((1/self.pre) + (1/self.eff))
        self.logloss = log_loss(tf.squeeze(y), scoresthr)

        stats = [{'stat': 'sig_in_test', 'value': len(y[y==1])},
                 {'stat': 'bkg_in_test', 'value': len(y[y==0])},
                 {'stat': 'accuracy', 'value': self.acc},
                 {'stat': 'precision', 'value': self.pre},
                 {'stat': 'efficiency', 'value': self.eff},
                 {'stat': 'f1score', 'value': self.f1s},
                 {'stat': 'test loss', 'value': self.logloss},
                 {'stat': 'threshold', 'value': self.min_thr}]
        fieldnames = ['stat', 'value']

        if save:
            with open(f'{self.outdir}/{self.modeldir}/nmodel_stats.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(stats)
            print('Stats Saved')

        
def main():

    TRAIN = True                   # Set to true if you want to train a new / existing model. Set to false if you only want to test a pre-trained model
    TEST = True                     # Set to true if you want to test the model being trained (or also test a pre-trained model if TRAIN is false --> must specify MODELDIR)
    FINETUNE = False                # Set to true if you want to load weights from a previous model to continue training --> must specify MODELDIR
    OUTDIR = 'reg_only_model_retest'           # Set to the name of the folder where you will save your models
    MODELDIR = '20240126--221615'    # If FINETUNE is true or Train is false and Test is true, MODELDIR must be specified as the dt_string of the model to be finetuned / tested or an error will occur
    TRAINDATA_DIR = '/storage/2/ek19824/ML_data/nosat_dataset/dataset_training.pickle'     # Provide the path to your training dataset
    TESTDATA_DIR = '/storage/2/ek19824/ML_data/nosat_dataset/dataset_testing.pickle'       # Provide the path to your testing dataset


    model = Model(OUTDIR)

    ### Train new model
    if FINETUNE is False and TRAIN is True: 
        model.new_model()
        train_data = prepare_dataset(TRAINDATA_DIR)
        model.fit_model(train_data)
        model.plot_training_curves(save=True, history=model.history)

    ### Continue training of previously trained model
    elif FINETUNE is True and TRAIN is True:
        assert MODELDIR is not None, 'Must provide the name of the specific model to finetune using option -md in command line\n(e.g. -md 20231108--120222)'
        model.load_model(MODELDIR)
        train_data = prepare_dataset(TRAINDATA_DIR)
        model.fit_model(train_data)
        model.plot_training_curves(save=True)


    ### Test Model
    if TEST is True:

        # Load previously trained model
        if TRAIN is False:      
            assert MODELDIR is not None, 'Must provide the name of the specific model to test using option -md in command line\n(e.g. -md 20231108--120222)'
            model.load_model(MODELDIR)

        test_data = prepare_dataset(TESTDATA_DIR)
        model.test_model(test_data, save=True, verbose=False)
        #model.plot_ROC(test_data['y'], save=True)
        #model.plot_confusion_matrix(test_data['y'], save=True)
        model.plot_regression_performance(test_data['ht'],save=True)
        print(f'R2: {model.reg_r2:.4f}')
        #print(f'Accuracy: {model.acc:.3f}')
        #print(f'Efficiency: {model.eff:.3f}')
        #print(f'Acceptance Threshold: {model.min_thr:.5f}')

    print(f'> Info: model {model.modeldir} done')
        

if __name__ == "__main__":
    main()

