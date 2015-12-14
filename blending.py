# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:20:21 2015

@author: lypan
"""
#%%
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import csv
import numpy as np
import pandas as pd
import random
import re
import time
import xgboost as xgb
#%%#################### original data
start_time = time.time()
# read train and test data from json
traindf = pd.read_json("train.json")
testdf = pd.read_json("test.json")


# data preprocessing
# 1. replace '-' with '_'
# 2. encode from unicode to ascii
# 3. transform from uppercase to lowercase
# 4. add a column for the clean string result
traindf['processed_ingredients'] = traindf.apply(lambda x: [re.sub('-', '_', i.encode('ascii', 'ignore').lower().strip()) for i in x['ingredients']], axis=1)
testdf['processed_ingredients'] = testdf.apply(lambda x: [re.sub('-', '_', i.encode('ascii', 'ignore').lower().strip()) for i in x['ingredients']], axis=1)
traindf['processed_ingredients_string'] = [','.join(z).strip() for z in traindf['processed_ingredients']]
testdf['processed_ingredients_string'] = [','.join(z).strip() for z in testdf['processed_ingredients']]


# Encode cuisine to integer(start from 0)
le = preprocessing.LabelEncoder()
le.fit(traindf['cuisine'])
traindf['cuisine_number'] = le.transform(traindf['cuisine'])
random.shuffle(traindf)
cuisine_label = traindf['cuisine_number']
#%%#################### data preprocessing
# Document term matrix
vectorizer = CountVectorizer()
dtm_train = vectorizer.fit_transform(traindf['processed_ingredients_string']).toarray()
dtm_test = vectorizer.transform(testdf['processed_ingredients_string']).toarray()


# tf-idf transforming
# tfidf_trans = TfidfTransformer()
# tfidf_train = tfidf_trans.fit_transform(dtm_train).toarray()
# tfidf_test = tfidf_trans.transform(dtm_test).toarray()
#%%####################level 0
clfs = [
    RandomForestClassifier(n_estimators = 100, criterion = 'gini'),
    ExtraTreesClassifier(n_estimators = 100 * 2, criterion = 'gini'),
    GradientBoostingClassifier(n_estimators = 100),
]
# clf1 = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, n_estimators=275, min_samples_split=3, criterion='gini', max_features=43, max_depth=None)
# # clf2 = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, n_estimators=150, min_samples_split=6, criterion='gini', max_features=40, max_depth=None)
# clf3 = ExtraTreesClassifier(max_features=6, n_estimators=100, criterion='gini', max_depth=None)
# # clf4 = ExtraTreesClassifier(max_features=1, n_estimators=70, criterion='entropy', max_depth=None)
# clf5 = LinearSVC(penalty='l2', C=0.2, tol=0.0001, dual=False)
# # clf6 = LinearSVC(penalty='l2', C=0.6, tol=0.01, dual=False)
# clf7 = LogisticRegression(penalty='l2', C=1, tol=0.0001, dual=False)
# clf8 = LogisticRegression(penalty='l2', C=8, tol=0.01, dual=False)
#%%
# Ready for cross validation
n_folds = 5;
skf = list(StratifiedKFold(cuisine_label, n_folds))
# Pre-allocate the data
blend_train = np.zeros((dtm_train.shape[0], len(skf))) # Number of training data x Number of classifiers
blend_test = np.zeros((dtm_test.shape[0], len(skf))) # Number of testing data x Number of classifiers
# For each classifier, we train the number of fold times (=len(skf))
for j, clf in enumerate(clfs):
    print 'Training classifier [%s]' % (j)
    blend_test_j = np.zeros((dtm_test.shape[0], len(skf))) # Number of testing data x Number of folds , we will take the mean of the predictions later
    for i, (train_index, cv_index) in enumerate(skf):
        print 'Fold [%s]' % (i)
        
        # This is the training and validation set
        X_train = dtm_train[train_index]
        Y_train = cuisine_label[train_index]
        X_cv = dtm_train[cv_index]
        Y_cv = cuisine_label[cv_index]
        
        clf.fit(X_train, Y_train)
        
        # This output will be the basis for our blended classifier to train against,
        # which is also the output of our classifiers
        blend_train[cv_index, j] = clf.predict(X_cv)
        blend_test_j[:, i] = clf.predict(X_test)
    # Take the mean of the predictions of the cross validation set
    blend_test[:, j] = blend_test_j.mean(1)
    
# Start blending!
bclf = LogisticRegression()
bclf.fit(blend_train, cuisine_label)
    
