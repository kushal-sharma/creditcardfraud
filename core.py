import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
import sklearn.cross_validation as cv
import seaborn as sns
import pickle
import time
import os

from keras.models import Sequential
from keras.layers import Dense, LSTM

def train_nn(model, xtrain, ytrain, niters = 100):
    losses = []
    xp, yp = xtrain[ytrain == 1], ytrain[ytrain == 1]
    xn, yn = xtrain[ytrain == 0], ytrain[ytrain == 0]
    
    Np = len(xp)
    Nn = len(xn)
    rp = np.arange(0, Np)
    rn = np.arange(0, Nn)
    yp = np.array([1 - yp, yp]).T
    yn = np.array([1 - yn, yn]).T
    for i in range(0, niters):
        idxp = np.random.choice(rp, size=20)
        idxn = np.random.choice(rn, size=10)
        batch_x = np.vstack((xp[idxp], xn[idxn]))
        batch_y = np.vstack((yp[idxp], yn[idxn]))
        batch = np.hstack((batch_x, batch_y))
        batch = np.random.permutation(batch)
        losses.append(model.train_on_batch(batch[:, :-2], batch[:, -2:]))
    return np.array(losses)

# Fit dataset imbalance bug
# Imbalanced dataset doesn't realy work
def get_nn_dense(nlayers, input_shape):
    model = Sequential()
    model.add(Dense(nlayers[0], activation='relu', input_shape=input_shape))
    for l in nlayers[1:]:
        model.add(Dense(l, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# TODO figure out how LSTM works
def get_nn_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(10, activation='tanh', input_shape=input_shape))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def evaluate(model, xtest, ytest, prob=False):
    ypred = model.predict(xtest)
    if prob:
        ypred = 1 - np.argmax(ypred, axis=1)
    tp = len(ypred[(ypred == 1) & (ytest == 1)])
    fp = len(ypred[(ypred == 1) & (ytest == 0)])
    tn = len(ypred[(ypred == 0) & (ytest == 0)])
    fn = len(ypred[(ypred == 0) & (ytest == 1)])
    return tp, fp, tn, fn

def save_scikit_model_with_loc(file_loc, model):
    with open(file_loc, 'wb') as f:
        pickle.dump(model, f)
        f.close()

def save_scikit_model(model_prefix : str, model):
    date_time = time.strftime("%d-%m-%Y_%H:%M:%S")
    file_loc  = os.path.join(os.getcwd(), model_prefix + '-' + date_time)
    save_scikit_model_with_loc(file_loc, model)

def apply_function(df, cols, fs, keys=None):
    summary_dict = {}
    for c in cols:
        for f in fs:
            f(df, c)

def get_mean(df, c):
    print("mean = " + str(df[c].mean()))

def get_extrema(df, c):
    print(df[df[c] == df[c].min()])
    print(df[df[c] == df[c].max()])

def f1score(model, X, y):
    ypred = model.predict(X)
    tp = len(y[(ypred == 1) & (y == 1)])
    fp = len(y[(ypred == 0) & (y == 0)])
    fn = len(y[(ypred == 0) & (y == 1)])
    pr = tp / (tp + fn)
    re = tp / (tp + fp)
    return 2 * pr * re / (pr + re)
    
