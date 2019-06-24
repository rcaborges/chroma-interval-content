import os, sys
import json

import numpy as np
import simplejson
import time
import matplotlib.pyplot as plt
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score

data = json.load(open("data.json"))

def kmeans_cluster(songs,i_max):
    clusterer = KMeans(n_clusters=i_max, random_state=10)
    cluster_labels = clusterer.fit_predict(songs)
    s_avg = silhouette_score(songs, cluster_labels)
    #print(s_avg)
    return cluster_labels, clusterer.cluster_centers_, clusterer,s_avg

def codewords(data,method,codes):
    counts = list(data.keys())
    print(len(counts))
    n_count = len(data.keys())
    matrix = np.zeros((12,1))
    cts = []
    ids = []
    
    for song in counts:
        ids.append(int(song))
        c1 = data[song][method]
        c1 = np.array(c1)
        matrix = np.hstack((np.array(matrix),c1))
    
    matrix = np.array(matrix)
    matrix = matrix[:,1:].T
    print(matrix.shape)
    
    labels,centroids,kmeans,s_avg = kmeans_cluster(matrix,codes)
    pop_cent = []
    #pop_cent.append(len(ds))
    labels = list(labels)
    for k in range(len(set(labels))):
        pop_cent.append(labels.count(k))
    del labels
    print(pop_cent)
    
    #### LOOP POR MUSICA
    vq = []
    codeMatrix = []
    ids_new = []
    labelsc = []
    for song in counts:
        #ids.append(int(country))
        c1 = data[song][method]
        c1 = np.array(c1)
        if c1.shape[1] > 0:
            ids_new.append(int(song))
            labelsc.append(data[song]["genre"])
            codewords = [] 
            mfcc_evol = kmeans.predict(c1.T)
            mybins = np.linspace(0, codes, codes+1)
            hist = np.histogram(mfcc_evol,mybins)
            hist = np.array(hist[0]/np.sum(hist[0]))
            hist = list(hist)
            codeMatrix.append(hist)
    cic = np.array(codeMatrix)
    print(cic.shape)
    from collections import Counter

    label_final = []
    for label in labelsc:
        if "Rock" in label: label_final.append("Rock")
        elif "Dance" in label: label_final.append("Dance")
        elif "R&B" in label: label_final.append("R&B")
        else: label_final.append(label)

    letter_counts = Counter(label_final)
    print(letter_counts)

    data_t = np.column_stack((cic,label_final))
    data_final = []
    grs = ["Rock","Slow","Dance"]
    for itt in data_t:
        if itt[-1] in grs: data_final.append(itt)
     
    #MAINTAIN ROCK AS 1 AND THE REST 0
    final_final = []
    for itt in data_final:
        iterit = []
        iterit.extend(itt[:-1])
        if itt[-1] == "Rock":iterit.append(1)
        else:iterit.append(0)
        final_final.append(iterit)

    return np.array(final_final)

def classify(data_final):
    NUM_THREADS = 5
    new_params = {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 15}
    xgb = XGBClassifier(**new_params, nthread=NUM_THREADS, seed=1, missing=np.nan)

    N_FOLD = 5
    scores = cross_val_score(xgb, data_final[:,:-1].astype(float), data_final[:,-1], cv=N_FOLD, scoring='f1_weighted')
    print("Fscore Weighted: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(xgb, data_final[:,:-1].astype(float), data_final[:,-1], cv=N_FOLD, scoring='f1_micro')
    print("Fscore Micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(xgb, data_final[:,:-1].astype(float), data_final[:,-1], cv=N_FOLD, scoring='f1_macro')
    print("Fscore Macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


codeMCIC = codewords(data,"CIC",4)
codeMDCF = codewords(data,"DCF",5)

classify(codeMCIC)
classify(codeMDCF)

