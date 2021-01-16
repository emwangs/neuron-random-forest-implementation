# Author Emily Wang
#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pandas as pd
import glob
import scipy
import sklearn as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection,metrics,tree
import anal_util as au  
from matplotlib import pyplot as plt
import operator
from matplotlib.colors import ListedColormap

## Graph the output using contour graph
def plotPredictions(max_feat, train_test, predict_dict, neurtypes, feature_order,epoch):
    #inputdf contains the value of a subset of features used for classifier, i.e., two different columns from df
    feature_cols = [feat[0] for feat in feature_order]
    inputdf = alldf[feature_cols[0:max_feat]]
    
    plt.ion()
    edgecolors=['k','none']
    feature_axes=[(i,i+1) for i in range(0,max_feat,2)]
    for cols in feature_axes:
        plt.figure()
        plt.title('Epoch '+str(epoch))
        for key,col in zip(train_test.keys(),edgecolors):
            predict = predict_dict[key]
            df = train_test[key][0]
            plot_predict = [neurtypes.index(p) for p in predict]
            plt.scatter(df[feature_cols[cols[0]]], df[feature_cols[cols[1]]], c=plot_predict,cmap=ListedColormap(['r', 'b']), edgecolor=col, s=20,label=key)
            plt.xlabel(feature_cols[cols[0]])
            plt.ylabel(feature_cols[cols[1]])
            plt.legend()


def plot_features(list_features,epochs,ylabel):
    plt.ion()
    objects = [name for name, weight in list_features]
    y_pos = np.arange(len(list_features))
    performance = [weight for name, weight in list_features]
    f = plt.figure(figsize=(6,4))

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.xlabel('Feature')
    plt.title(ylabel+' over '+epochs+' epochs')

## Runs cluster analysis on prepped data
def run_cluster_analysis(param_values, labels, num_features, alldf, epoch, MAXPLOTS):
    #select a random subset of data for training, and use the other part for testing

    df_values_train, df_values_test, df_labels_train, df_labels_test = model_selection.train_test_split(param_values, labels, test_size=0.33)
    train_test = {
      'train': (df_values_train, df_labels_train), 
      'test': (df_values_test, df_labels_test)
    }

    #number of estimators (n_estim) is number of trees in the forest
    #max_feat is the number of features to use for classification
    #Empirical good default value is max_features = sqrt(num_features) for classification tasks
    max_feat = int(np.ceil(np.sqrt(num_features)))
    n_estim = 10
    rtc = RandomForestClassifier(n_estimators=n_estim, max_features=max_feat)

    #Builds the random forest (does the training)
    rtc.fit(df_values_train, df_labels_train)

    #Calculate a score, show the confusion matrix
    predict_dict = {}
    for nm,(df,label) in train_test.items():
        predict = rtc.predict(df)
        predict_dict[nm] = predict

    #Evauate the importance of each feature in the classifier
    #The relative rank (i.e. depth) of a feature used as a decision node in a tree can be used to assess the relative importance of that feature with respect to the predictability of the target variable. 
    feature_order = sorted({feature : importance for feature, importance in zip(list(df_values_train.columns), list(rtc.feature_importances_))}.items(), key=operator.itemgetter(1), reverse=True)
    
    # 3d, plot and print the predictions of the actual data -- you can do this if # of epochs is low
    if epoch <= MAXPLOTS:
        plotPredictions(max_feat, train_test, predict_dict, neurtypes, feature_order,epoch)
    return feature_order[0:max_feat], max_feat

## Sets up data files for Cluster Analysis
## param path_root: directory and root file name containing the set of .npz files. ex. "/path/fileroot"
## param neurtypes: array containing the neuron types you want to identify between
## param tile: percentage of best fit neurons desired
## param num_fits: how many of each fit for classification of just a few of best fit neurons
def set_up_df(neurtypes, path_root, tile=0.005, num_fits=None): 
    #set of data files from parameter optimization
    pattern = path_root+'*.npz'
    
    #if small=True, use num_fits from each optimization, else, use %tile
    small = True

    #retrieve data files - sort the files by which neurtype
    file_names = glob.glob(pattern)
    group_names = {key:[f for f in file_names if key in f] for key in neurtypes}
    
    if len(file_names)==0:
        print('no files found by searching for', pattern)
    
    ##### process all examples of each type, combine into dict of data frames and then one dataframe
    df_list = {}
    df_list_of_lists = {} 
    for neur in neurtypes:
        df_list[neur], df_list_of_lists[neur] = au.combined_df(group_names[neur], tile, neur) #this method joins multiple dataframes
        #df_list[neur] is a DATAFRAME
        #df_list_of_lists[neur] is a LIST OF DATAFRAMES (1 dataframe per npz file)
        
    alldf = pd.concat([df for df in df_list.values()])
    print('all files read. Neuron_types: ', pd.unique(alldf['neuron']), 'df shape', alldf.shape,'columns', alldf.columns,'files', pd.unique(alldf['cell']),'\n')
    
    ####create smaller df using just small and same number of good fits from each neuron
    min_samples = np.min([n.shape[0] for vals in df_list_of_lists.values() for n in vals])
    if num_fits:
        num_samples=min(min_samples, num_fits)
    else:
        num_samples=min_samples
    smalldf_list = {neur:[] for neur in neurtypes}

    for neur in neurtypes:
        for i in range(len(df_list_of_lists[neur])):
            smalldf_list[neur].append(df_list_of_lists[neur][i][-num_samples:])
    
    if num_fits:
        alldf = pd.concat([df for dfset  in smalldf_list.values() for df in dfset])
        
    #exclude entire row (observation) if Nan is found
    alldf = alldf.dropna(axis=1)
    
    #identify fitness columns and number of features (parameter values)
    fitnesses = [col for col in alldf.columns if 'fitness' in col]
    chan_params = [col for col in alldf.columns if 'Chan' in col]
    num_features = len(alldf.columns)-len(fitnesses)

    print('new shape', alldf.shape,'fitnesses: ', len(fitnesses), 'params', num_features)

    #create dataframe with the 'predictor' parameters - conductance and channel kinetics
    #exclude columns that containing neuron identifier or fitness values, include the total fitness
    exclude_columns = fitnesses + ['neuron','neurtype','junction_potential', "model", "cell", 'total'] #total? ['neuron','neurtype','junction_potential']
    param_columns = [column for column in list(alldf.columns) if column not in exclude_columns]
    param_values = alldf[param_columns]

    #labels contains the target values (class labels) of the training data
    labels = alldf['neuron']
    
    return (param_values, labels, num_features, alldf) 


############ MAIN ############# 
# Define parameters  
epochs = 10 #100
neurtypes = ['Npas', 'proto'] 
path_root = 'opt_output/temeles_gpopt_output/'
tile = 0.005 
num_fits = 10 

# Set MAXPLOTS to zero to suppress plotting graphs
MAXPLOTS = 3

# Read in all npz files, select top tile% of model fits, put into pandas dataframe
param_values, labels, num_features, alldf = set_up_df(neurtypes,path_root,tile, num_fits)

# Do cluster analysis 
# Top 8 features & their weights in each epoch are cumulatively summed in collectionBestFeatures = {feature: totalWeightOverAllEpochs}                                                                                                  
# Top 1 feature in each epoch is stored in collectionTopFeatures = {feature: numberOfTimesAsTopFeatureOverAllEpochs}

collection_best_features = {}
collection_top_features = {}
for epoch in range(0, epochs):
    features, max_feat = run_cluster_analysis(param_values, labels, num_features, alldf, epoch, MAXPLOTS)
