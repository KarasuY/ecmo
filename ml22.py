#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import random
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score

#Load data from csv file
def loadCsv(filename):
    lines = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
    next(lines)
    dataset = []
    for row in lines:
        dataset.append([float(x) for x in row])
    return dataset

def main():
    filename = 'readmission.csv'
    dataset = np.array(loadCsv(filename))
    seed = 42
    
    X = dataset[:, :67]
    y = dataset[:, 68]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)
    print('Split '+str(len(dataset))+' rows into train = '+str(len(X_train))+' and test = '+str(len(X_val))+' rows.\n')
    
    c = GaussianNB()
    y_pred = c.fit(X_train, y_train).predict(X_val)
    scores = f1_score(y_val, y_pred, average=None)
    for i in range(len(scores)):
        print('F1 (Class '+str(i)+'): '+str(scores[i]))

main()

