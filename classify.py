import os, sys
import json

import numpy as np
import simplejson
from sklearn.metrics import jaccard_similarity_score
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.vq import vq, kmeans, whiten
import scipy

from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from sklearn import preprocessing

data = json.load(open("data.json"))

def plot_histogram(anos):
    fig,ax = plt.subplots()
    plt.bar( range(12), anos)
    labels = ['-5','-4','-3','-2','-1','1','2','3','4','5','6']
    indexes = np.arange(len(anos))
    plt.xticks(indexes, labels)
    #fig.savefig('histogram.png', bbox_inches='tight')
    plt.show()

def kmeans_cluster(songs,i_max):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_samples, silhouette_score

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
    
    ### CODEWORDS
#    codes = 2
#    avgmax = 0
#    for i in np.arange(2,100,1):
#        labels,centroids,kmeans,s_avg = kmeans_cluster(matrix,i)
#        print(i,s_avg,codes)
#        if s_avg > avgmax: 
#            avgmax = s_avg
#            codes = i
#    print("MAX: ",codes,avgmax)
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
    #grs = list(letter_counts)
    grs = ["Rock","Slow","Dance","Country"]
    #grs = ["Rock","Slow"]
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
    #split = int(0.8*data_final.shape[0])
    #
    #X_train = data_final[:split,:-1].astype(np.float)
    #X_test = data_final[split:,:-1].astype(np.float)
    #y_train = data_final[:split,-1].astype(int)
    #y_test = data_final[split:,-1].astype(int)
    #
    NUM_THREADS = 5
    #print('Training model...')
    ##new_params = {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 15, 'n_estimetors': 100}
    new_params = {'gamma': 0, 'learning_rate': 0.2, 'max_depth': 15}
    xgb = XGBClassifier(**new_params, nthread=NUM_THREADS, seed=1, missing=np.nan)
    #xgb = XGBClassifier()
    #xgb.fit(X_train, y_train)
    #print('Predicting...')
    #y_pred = xgb.predict(X_test)
    #report = classification_report(y_test, y_pred)
    #print(report)
    #print('Predicting...')
    #y_pred = xgb.predict(X_train)
    #report = classification_report(y_train, y_pred)
    #print(report)
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score

    N_FOLD = 5
    scores = cross_val_score(xgb, data_final[:,:-1].astype(float), data_final[:,-1], cv=N_FOLD, scoring='f1_weighted')
    print("Fscore Weighted: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(xgb, data_final[:,:-1].astype(float), data_final[:,-1], cv=N_FOLD, scoring='f1_micro')
    print("Fscore Micro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(xgb, data_final[:,:-1].astype(float), data_final[:,-1], cv=N_FOLD, scoring='f1_macro')
    print("Fscore Macro: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

def plot_feature(data):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    RS = 20150101
    pca = PCA(n_components=7)
    #X_t = pca.fit(data[:,:-1].astype(np.float)).transform(data[:,:-1].astype(np.float))
    X_t = TSNE(random_state=RS).fit_transform(data[:,:-1].astype(np.float))
    #print(np.sum(pca.explained_variance_ratio_)) 

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(data[:,-1])
    labels = le.transform(data[:,-1])
    plt.figure(figsize=(10, 8))
    plt.scatter(X_t[:,0], X_t[:,1],c=labels,cmap='prism')  # plot points with cluster dependent colors
    #plt.scatter(X_t[:,0], X_t[:,1],cmap='prism')  # plot points with cluster dependent colors
    i=0
    for txt in data[:,-1]:
        plt.annotate(str(txt),(X_t[i,0],X_t[i,1]))
        i=i+1
    plt.show()


codeMCIC = codewords(data,"CIC",4)
codeMDCF = codewords(data,"DCF",5)

#arr = np.arange(codeMCIC.shape[0])
#np.random.shuffle(arr)
#codeMCIC = codeMCIC[arr,:]
#codeMDCF = codeMDCF[arr,:]

classify(codeMCIC)
classify(codeMDCF)

#plot_feature(codeMCIC)
#plot_feature(codeMDCF)
