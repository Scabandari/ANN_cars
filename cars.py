#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 16:28:37 2019

@author: ryan
Refs: https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
& https://www.udemy.com/deeplearning/
"""

import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
r = requests.get(url)
#r.text
lines = r.text.split("\n")
data = []
for line in lines:
    words = line.split(',')
    data.append(words)
    
cols = ['price', 
           'maintenance_cost',
           'num_doors',
           'num_persons',
           'lug_boot',
           'safety', 
           'label']
x_cols = cols[:-1]
         
df = np.array(data)
df = pd.DataFrame(data=data, columns=cols)
print(df.head())
print(df.tail())
df.drop(df.index[-1], inplace=True)

possible_datatypes = {}
for col in cols:
    possible_datatypes[col] = df[col].unique()
for col in cols: 
    print(possible_datatypes[col])
df = df.replace('more', 5)
df = df.replace('5more', 5)
df = df.replace('2', 2)
df = df.replace('3', 3)
df = df.replace('4', 4)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y = np_utils.to_categorical(y)

# To ensure the proper dummy variable is dropped I'm separating 
# the columns that need to be one hot encoded, dropping the variable
# and I'll have to rejoin the df's at the end
data_frames = []
change_cols = [0, 1, 4, 5]

# for each column that needs to --> One hot encoding
# isolate the column + a column of zeros in a df (label encoder expects > 1 cols)
# change to 2D numpy array
# get one hot encoding
# add to array to be joined later
for i in change_cols:
    second = 4 if i == 5 else i+1 
    df_ = df.copy(deep=True)
    df_ = df_[[cols[i], cols[second]]]
    df_[cols[second]] = 0
    X_ = df_.iloc[:, :].values
    #print(df_.head())
    labelencoder = LabelEncoder()
    X_[:, 0] = labelencoder.fit_transform(X_[:, 0])
    #df_n = pd.DataFrame(X_)
    #print(df_n.head())
    onehotencoder = OneHotEncoder(categorical_features = [0])
    X_ = onehotencoder.fit_transform(X_).toarray()
    X_ = X_[:, 1:-1]
    #df_n = pd.DataFrame(X_)
    data_frames.append(pd.DataFrame(X_))
    #print(df_n.head())
    #data_frames.append(pd.DataFrame(X_))
    
# provide correct column names for df's in data_frames[]
for index, data_fr in enumerate(data_frames[1:]):
    start = data_frames[index].columns[-1] + 1
    #print("Index: {}".format(index))
    #print("start: {}".format(start))
    new_cols = []
    cols_ = len(data_fr.columns)
    for i in range(start, start + cols_):
        new_cols.append(i)
    #print("Adding cols: {}".format(new_cols))
    data_fr.columns = new_cols
    
df_f = pd.concat( data_frames, axis=1, sort=False)
df_f['num_doors'] = df['num_doors']
df_f['num_persons'] = df['num_persons']

X = df_f.values

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
# Part 2 Making the ANN!

# Initialize the ANN
# define as a sequence of layers
def baseline_model():
    classifier = Sequential()
    # input layer and first hidden layer
    classifier.add(Dense(input_dim=12, activation="relu", units=6, kernel_initializer="uniform"))
    # adding the second hidden layer
    classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

    # adding the output layer
    classifier.add(Dense(units=4, activation="softmax",  kernel_initializer="uniform"))
    # activation was sigmoid previously, that's better for binary outputs?
    # compliling the ANN
    # adam is a type of stochastic gradient decent
    classifier.compile( optimizer='adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])
    return classifier
    
estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


    