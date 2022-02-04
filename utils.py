# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 10:43:55 2022

@author: tahamansouri
"""
import numpy as np
import torch
import pandas as pd


def dataset(paper):
    dataSets=[]
    dataSets.append(pd.read_csv('dataset/machine-1.csv'))
    dataSets.append(pd.read_csv('dataset/machine-2.csv'))
    dataSets.append(pd.read_csv('dataset/machine-3.csv'))
    dataSets.append(pd.read_csv('dataset/machine-4.csv'))
    return dataSets

def data_preprocessing(dset, colname,T, D,errN,errThresh,ratio):
    data=dset[colname].values
    data=data[(len(data)-(len(data)//T)*T):]
    X=[]
    Y=[]
    Y_real=[]
    for t in range(len(data) - 2*T):
        x = data[t:t+T]
        X.append(x)
        y = data[t+T:t+T+T]
        Y_real.append(y.copy())
        Y.append(y)
        
    for g in range(len(Y)):
        mid=(Y[g]>errThresh).sum()
        if mid>=errN:
            Y[g]=1
        else:
            Y[g]=0
    X_train=X[:int(ratio*len(X))]
    X_test=X[int(ratio*len(X)):]
    Y_train=Y[:int(ratio*len(Y))]
    Y_test=Y[int(ratio*len(Y)):]
    Y_real_train = Y_real[:int(ratio*len(Y_real))]
    Y_real_test = Y_real[int(ratio*len(Y_real)):]
    return X_train,Y_train,X_test,Y_test,Y_real_train,Y_real_test

def data_prepare(data,colname,T, D,errN,errThresh, ratio):
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    Y_real_train=[]
    Y_real_test=[]
    for dset in data:
        A,B,C,D,E,F = data_preprocessing(dset, 'Value',T, D,errN,errThresh,ratio)
        X_train+=A
        Y_train+=B
        X_test+=C
        Y_test+=D
        Y_real_train+=E
        Y_real_test+=F
    X_train = np.array(X_train).reshape(-1, T, 1) # Now the data should be N x T x D
    Y_train = np.array(Y_train)
    X_test = np.array(X_test).reshape(-1, T, 1) # Now the data should be N x T x D
    Y_test = np.array(Y_test)
    Y_real_train = np.array(Y_real_train).reshape(-1, T)
    Y_real_test = np.array(Y_real_test).reshape(-1, T)
    return X_train,Y_train,X_test,Y_test,Y_real_train,Y_real_test

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.as_tensor(ID)
        y = self.labels[index]

        return X, y