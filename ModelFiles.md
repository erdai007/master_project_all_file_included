What's in Model File
=====

In a typical model file, there are the following components

-```KerasModel```
-```KerasModel.h5```
-```MetricsKerasModel.json```
-```nconfusion_matrix.jpg```
-```nmodel_stats.csv```
-```ntraining_curve.jpg```
-```reg_comparison.png```
-```roc.png```
-```predictions.pickle```

Model itself
----

All essential information for defining the model, including its parameters trained and its architecture, are stored in ```KerasModel``` and ```KerasModel.h5```

Metrics in training
----

The performance of the model during the training, e.g., the behaviour of loss function of the model in each epoch of training, is stored as a json file in ```MetricsKerasModel.json```, and is plotted in ```ntraining_curve.jpg```.

```prediction.pickle```
----

This is the file which takes the image inputs and returns predicted values, $H_{T} $, jets' visible energies, and $y$, classification scores. They are used for produce roc curves and regression plot using scripts in ```plotting_and_auxiliary_scripts```.

Performance 
----

Classification performance was plotted in ```roc.png``` for roc curve(AUC printed), and ```nconfusion_matrix.jpg``` for the confusion matrix observed at $10$ kHz triggering rates. Regression performance was plotted in ```reg_comparison.png```, with $R^2$ included.

Stats(Not applicable for regression only model.)
----

All stats, including AUC, $R^2 $, signal efficiency at $10$ kHz, were stored in ```nmodel_stats.csv```.