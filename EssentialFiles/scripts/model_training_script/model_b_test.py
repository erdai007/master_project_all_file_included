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


def create_keras_model(nBinsX, nBinsY,model_loss_weight=[1.0,1.0],regression_act_type='relu'):
        
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

    dense1 = Dense(32, activation='relu')(flat)
    dense2 = Dense(16, activation='relu')(dense1)
    dense3 = Dense(8, activation='relu')(dense2)
    bn_r = BatchNormalization(axis=-1)(dense3)
    regression_output = Dense(1, activation=regression_act_type,name="regression_output")(bn_r)
    
    merge = Concatenate()([flat,regression_output])
    dense4 = Dense(50,activation='relu')(merge)
    dense5 = Dense(25,activation='relu')(dense4)
    dense6 = Dense(8,activation='relu')(dense5)
    bn_c = BatchNormalization(axis=-1)(dense6)

    classification_output = Dense(1, activation='sigmoid', name="classification_output")(bn_c)
    
    

    # Define the model with both the image and additional input as inputs
    model = tf.keras.models.Model(inputs=img_input, outputs=[classification_output,regression_output ])
    
    # Defining loss type
    regression_loss = 'mean_squared_error'
    classification_loss='binary_crossentropy'
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(loss={'classification_output':classification_loss,'regression_output':regression_loss}, optimizer=opt, metrics={'classification_output':'accuracy','regression_output':regression_loss})
    
    model.compile(loss={'classification_output':classification_loss,'regression_output':regression_loss}, loss_weights={'classification_output':model_loss_weight[0],'regression_output':model_loss_weight[1]}, optimizer=opt, metrics={'classification_output':['accuracy',tf.keras.metrics.SensitivityAtSpecificity(1-0.00025, num_thresholds=10000)],'regression_output':regression_loss})
    
    #model.compile(loss={'classification_output':classification_loss,'regression_output':regression_loss}, optimizer=opt, metrics=[tf.keras.metrics.SensitivityAtSpecificity(1-0.00025, num_thresholds=10000),{'classification_output':'accuracy','regression_output':regression_loss}  ])
    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', tf.keras.metrics.SensitivityAtSpecificity(1-0.00025, num_thresholds=10000)])
    
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
   
    def __init__(self, outdir,model_loss_weight=[1.0,1.0],regression_act_type='relu') -> None:
        self.outdir = outdir
        self.loss_weight = model_loss_weight
        self.clas_weight = self.loss_weight[0]
        self.reg_weight = self.loss_weight[1]
        self.reg_act = regression_act_type

        pass


    def new_model(self):
        self.model = create_keras_model(78, 78, model_loss_weight=self.loss_weight,regression_act_type=self.reg_act)
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

        self.history = self.model.fit(train_data['x'], [train_data['y'],train_data['ht']],
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


    def print_history(self, history=False):
        if not history:
            with open(f"{self.outdir}/metrics_{self.model_name}.json", 'r') as file:
                self.history = json.load(file)
                file.close()
        
        print(self.history.history.keys())
        print(self.history.params)
    
    
    def plot_training_curves(self, history=False, save=False):

        if not history:
            with open(f"{self.outdir}/{self.modeldir}/metrics_{self.model_name}.json", 'r') as file:
                self.history = json.load(file)
                file.close()

        #plt.figure(figsize=(800,800), dpi=200)
        fig, ((ax_acc, ax_cl_loss), (ax_reg_loss, ax_SAS)) = plt.subplots(2,2, sharey=False)
        fig.set_size_inches(6.0,6.7)
        fig.set_dpi(200)

        ax_acc.plot(self.history.history['classification_output_accuracy'])
        ax_acc.plot(self.history.history['val_classification_output_accuracy'])
        ax_acc.set_ylabel('classification accuracy')
        ax_acc.set_xlabel('epoch')
        ax_acc.legend(['train', 'validation'], loc='lower left')

        ax_cl_loss.plot(self.history.history['classification_output_loss'])# self.history.history['lo']
        ax_cl_loss.plot(self.history.history['val_classification_output_loss'])
        ax_cl_loss.set_ylabel('classification loss')
        ax_cl_loss.set_xlabel('epoch')
        ax_cl_loss.legend(['train', 'validation'], loc='upper left')
        
        ax_reg_loss.plot(self.history.history['regression_output_mean_squared_error'])
        ax_reg_loss.plot(self.history.history['val_regression_output_mean_squared_error'])
        ax_reg_loss.set_ylabel('regression error')
        ax_reg_loss.set_xlabel('epoch')
        ax_reg_loss.legend(['train','validation'],loc='lower right')

        ax_SAS.plot(self.history.history['classification_output_sensitivity_at_specificity'])
        ax_SAS.plot(self.history.history['val_classification_output_sensitivity_at_specificity'])
        ax_SAS.set_ylabel('SAS')
        ax_SAS.set_xlabel('epoch')
        ax_SAS.legend(['train', 'validation'], loc='upper left')

        fig.suptitle(f"Training Curves")  
        fig.tight_layout()
        if save:
          plt.savefig(f"{self.outdir}/{self.modeldir}/ntraining_curves.jpg")
          print('Training Curves Plotted')
        plt.show()


    def test_model(self, test_data, save=True, verbose=False):
        
        if verbose: 
          print(self.model.summary())
          
        self.scores = self.model.predict(test_data['x'])[0]
        #print(np.unique(self.scores))
        self.regress_val = self.model.predict(test_data['x'])[1]
        #print(self.regress_val)
        idx = np.array(tf.squeeze(test_data['idx']))
        sorted_idx = np.argsort(idx)
        results = {'idx': idx[sorted_idx], 'score': self.scores[sorted_idx], 'ht':self.regress_val[sorted_idx],'label': np.array(test_data['y'])[sorted_idx]}

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
      self.auc=roc_auc_score(list(map(int, list(y))), self.scores)
      self.min_thr = thresholds[fpr<10][-1]
      self.eff = tpr[thresholds==self.min_thr][0]
      plt.figure(dpi=200) 
      plt.plot(fpr,tpr,label='CNN',linestyle='-', color='orange', linewidth=3.0)
      #plt.plot(tpr_ht,fpr_ht,label='QuadJetHT (no pT selection)',linestyle='-')
      
      plt.text(2*10**3,0.1,"auc={:.5f}".format(self.auc))
      
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
        scoresthr[scoresthr>=1.5]=1.49999
        scoresthr[scoresthr<=-0.5]=-0.5
        #scoresthr = np.array(self.scores)
        print(np.unique(scoresthr.round()))
        matrix_confusion = confusion_matrix(list(map(int, list(y))), scoresthr.round())
        print(matrix_confusion)
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
        
    def Ht_Trigger_rate_plot(self,y,ht,sample_num=50,save=True):
        Ht_range, Ht_diff= np.linspace(0.0,1.0,sample_num+1,retstep=True)
        Ht_upper = Ht_range[1:]
        Ht_lower = Ht_range[:-1]
        Ht_interval = np.transpose(np.vstack((Ht_lower,Ht_upper)))
        Ht_midpoint = (Ht_upper + Ht_lower)/2
        
        Ht_predict=self.regress_val.flatten()
        Ht_true=ht.numpy().flatten()
        
        scoresthr = (np.array(self.scores - (self.min_thr - 0.5))).flatten()
        scoresthr[scoresthr>=1.5]=1.49999
        scoresthr[scoresthr<=-0.5]=-0.5
        y_predict = scoresthr.round()
        y_true = np.array(list(map(int, list(y)))).flatten()
                
        trigg_rate = []
        trigg_rate_ideal = []
        
        iteration_count=0
        
        for i in Ht_interval.tolist():
            iteration_count+=1
            
            lower_bound=i[0]
            upper_bound=i[1]
            y_pred_Ht_in_range= y_predict[(Ht_predict >=lower_bound) & (Ht_predict <= upper_bound)]
            y_true_Ht_in_range= y_true[(Ht_true >= lower_bound) & (Ht_true <= upper_bound)]
            
            Num_pred_in_range= y_pred_Ht_in_range.size
            Num_ideal_in_range= y_true_Ht_in_range.size
            trigg_num_pred_in_range= y_pred_Ht_in_range[y_pred_Ht_in_range==1].size
            trigg_num_ideal_in_range= y_true_Ht_in_range[y_true_Ht_in_range==1].size
            
            
            if Num_pred_in_range!= 0:
                trigg_rate.append(trigg_num_pred_in_range / Num_pred_in_range)
            elif len(trigg_rate)==0:
                trigg_rate.append(0.0)
                #print('0 trigger rate detected')
            else:
                trigg_rate.append(trigg_rate[-1])
                #print('ending trigger detected')
                
            if Num_ideal_in_range!= 0:
                trigg_rate_ideal.append(trigg_num_ideal_in_range / Num_ideal_in_range)
            elif len(trigg_rate_ideal)==0:
                trigg_rate_ideal.append(0.0)
            else:
                trigg_rate_ideal.append(trigg_rate_ideal[-1])
            
            # trigg_rate.append(trigg_num_pred_in_range / Num_pred_in_range) if Num_pred_in_range>0 else trigg_rate.append(trigg_rate[-1])
            # trigg_rate_ideal.append(trigg_num_ideal_in_range / Num_ideal_in_range) if Num_ideal_in_range>0 else trigg_rate_ideal.append(trigg_rate_ideal[-1])
            #print(trigg_rate) if i[1]<=10*Ht_diff else 0
            #end for###
        
        #print(trigg_rate)
        Trigg_rate_near_zero= (np.array(trigg_rate_ideal)[np.array(trigg_rate_ideal)>0])[0]
        Ht_low_Trigg_rate= np.min(Ht_midpoint[np.array(trigg_rate)>= 1.2*Trigg_rate_near_zero ] )
        if np.max(np.array(trigg_rate))<=0.9*np.max(np.array(trigg_rate_ideal)):#Only if fitting is
            Ht_high_Trigg_rate= 1.0
        else:
            Ht_high_Trigg_rate= np.min(Ht_midpoint[np.array(trigg_rate)>=0.99*np.max(np.array(trigg_rate_ideal))])
        
        self.Ht_low_trigg_rate=Ht_low_Trigg_rate
        self.Ht_high_trigg_rate=Ht_high_Trigg_rate
        self.Ht_sample_width=Ht_diff/2
        
        print(iteration_count)
        plt.figure(figsize=(10,8),dpi=200)     
        plt.plot(Ht_midpoint,trigg_rate,label='CNN',linestyle='-',color='blue')
        plt.plot(Ht_midpoint,trigg_rate_ideal,label='Ideal trigger rate',linestyle='-',color='red')
        
        plt.xlabel('Ht')
        plt.ylabel('Trigger rate')
        plt.xlim(-0.03,1.2)
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.title(f'Ht sample width: {Ht_diff/2}',loc='center')
        if save:
          plt.savefig(f'{self.outdir}/{self.modeldir}/Ht_trig_rate_plot.jpg')
          print('Trigger rate against Ht Plotted')
            
            
        return

    
    def save_stats(self,y, save=False):
        stats = [{'stat': 'sig_in_test', 'value': len(y[y==1])},
                 {'stat': 'bkg_in_test', 'value': len(y[y==0])},
                 {'stat': 'accuracy', 'value': self.acc},
                 {'stat': 'precision', 'value': self.pre},
                 {'stat': 'efficiency', 'value': self.eff},                 
                 {'stat': 'auc', 'value': self.auc},
                 {'stat': 'f1score', 'value': self.f1s},
                 {'stat': 'test loss', 'value': self.logloss},
                 {'stat': 'threshold', 'value': self.min_thr},
                 {'stat': 'regression_R^2', 'value': self.reg_r2},
                 {'stat':'classification loss weight','value':self.clas_weight},
                 {'stat':'regression loss weight','value':self.reg_weight},
                 {'stat':'regression activation type', 'value':self.reg_act},
                 {'stat':'Ht trigger rate quantities','value':'values'},
                 {'stat':'plot sample width','value':self.Ht_sample_width},
                 {'stat':'Ht limit with low triggering rate','value':self.Ht_low_trigg_rate},
                 {'stat':'Ht limit with high triggering rate','value':self.Ht_high_trigg_rate}]
        fieldnames = ['stat', 'value']

        if save:
            with open(f'{self.outdir}/{self.modeldir}/nmodel_stats.csv', 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(stats)
            print('Stats Saved')        

        
def main():

    TRAIN = False                   # Set to true if you want to train a new / existing model. Set to false if you only want to test a pre-trained model
    TEST = True                     # Set to true if you want to test the model being trained (or also test a pre-trained model if TRAIN is false --> must specify MODELDIR)
    FINETUNE = False                # Set to true if you want to load weights from a previous model to continue training --> must specify MODELDIR
    OUTDIR = 'model_b_layer_size_test'           # Set to the name of the folder where you will save your models
    MODELDIR = '20240228--204050'    # If FINETUNE is true or Train is false and Test is true, MODELDIR must be specified as the dt_string of the model to be finetuned / tested or an error will occur
    TRAINDATA_DIR = '/storage/2/ek19824/ML_data/nosat_dataset/dataset_training.pickle'     # Provide the path to your training dataset
    TESTDATA_DIR = '/storage/2/ek19824/ML_data/nosat_dataset/dataset_testing.pickle'       # Provide the path to your testing dataset


    clas_weight=0.5
    model = Model(OUTDIR,model_loss_weight=[clas_weight,1.0-clas_weight],regression_act_type='relu')

    ### Train new model
    if FINETUNE is False and TRAIN is True: 
        model.new_model()
        train_data = prepare_dataset(TRAINDATA_DIR)
        model.fit_model(train_data)
        #model.print_history(history=model.history)
        model.plot_training_curves(save=True, history=model.history)

    ### Continue training of previously trained model
    elif FINETUNE is True and TRAIN is True:
        assert MODELDIR is not None, 'Must provide the name of the specific model to finetune using option -md in command line\n(e.g. -md 20231108--120222)'
        model.load_model(MODELDIR)
        train_data = prepare_dataset(TRAINDATA_DIR)
        model.fit_model(train_data)
        model.plot_training_curves(save=True, history=model.history)


    ### Test Model
    if TEST is True:

        # Load previously trained model
        if TRAIN is False:      
            assert MODELDIR is not None, 'Must provide the name of the specific model to test using option -md in command line\n(e.g. -md 20231108--120222)'
            model.load_model(MODELDIR)

        test_data = prepare_dataset(TESTDATA_DIR)
        model.test_model(test_data, save=True, verbose=False)
        model.plot_ROC(test_data['y'], save=True)
        model.plot_regression_performance(test_data['ht'],save=True)
        model.plot_confusion_matrix(test_data['y'], save=True)
        model.Ht_Trigger_rate_plot(y=test_data['y'],ht=test_data['ht'],save=True,sample_num=100)
        model.save_stats(test_data['y'],save=True)


        print(f'Accuracy: {model.acc:.3f}')
        print(f'Efficiency: {model.eff:.3f}')
        print(f'Acceptance Threshold: {model.min_thr:.5f}')
        print(f'AUC: {model.auc:.3f}')
        print(f'R2: {model.reg_r2:.4f}')

    print(f'> Info: model {model.modeldir} done')
        

if __name__ == "__main__":
    main()

