# ------------- Author -------------#
# Samreen A. Khan

# ------------- Description ------------#
# This project is about multi-class classification with the Keras Deeep learning Library

import numpy
import pandas

from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# Setting the reproducibility value
seed = 7
numpy.random.seed(seed)

# Loading the dataset
dataframe = pandas.read_csv("iris.csv", header = None)
dataset = dataframe.values

X = dataset[:,0.4].astype(float)
Y = dataset[:,4]

# Creating baseline model
def baseline_model():
    model = Sequential()
    model.add(Dense(8, imput_dim = 4, activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

estimator = KerasClassifier(build_fn = baseline_model, epochs = 200, batch_size = 5, verbose = 0)

# Evaulating the model
kfold = kFold(n_splits = 10, shuffle = True, random_state = seed)

# Getting the results
results = cross_val_score(estimator, X, dummy_y, cv = kfold)
print ("Baseline = %.2f%% (%.2f%%)") % (results.mean()*100, results.std()*100)
