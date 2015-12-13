# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:00:35 2015

@author: liangyupan
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from operator import itemgetter
from sklearn import cross_validation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint
from sklearn.naive_bayes import MultinomialNB
from time import time
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import re
import csv
from math import sqrt
#%%

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


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
cuisine_label = traindf['cuisine_number']


# Document term matrix
vectorizer = CountVectorizer(max_features = 2000)
vectorizer.fit_transform(traindf['processed_ingredients_string'])
dtm_train = vectorizer.transform(traindf['processed_ingredients_string']).toarray()
dtm_test = vectorizer.transform(testdf['processed_ingredients_string']).toarray()


# Tf–idf term
tfidf_trans = TfidfTransformer()
tfidf_train = tfidf_trans.fit_transform(dtm_train).toarray()
tfidf_test = tfidf_trans.fit_transform(dtm_test).toarray()

#%%
#clf = RandomForestClassifier()
## specify parameters and distributions to sample from
#param_dist = {"n_estimators": [50, 75, 100],
#              "max_depth": [3, None],
#              "max_features": sp_randint(57, 69),
#              "min_samples_split": sp_randint(1, 11),
#              "min_samples_leaf": sp_randint(1, 11),
#              "bootstrap": [True, False],
#              "criterion": ["gini", "entropy"]}
#
## run randomized search
#n_iter_search = 20
#random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
#                                   n_iter=n_iter_search)
#
## DTM
#start = time()
#random_search.fit(dtm_train, cuisine_label)
#print("RandomizedSearchCV took %.2f seconds for %d candidates"
#      " parameter settings." % ((time() - start), n_iter_search))
#report(random_search.grid_scores_)
#
## TF-IDF
#start = time()
#random_search.fit(tfidf_train, cuisine_label)
#print("RandomizedSearchCV took %.2f seconds for %d candidates"
#      " parameter settings." % ((time() - start), n_iter_search))
#report(random_search.grid_scores_)

#%%
clf = MultinomialNB()
# specify parameters and distributions to sample from
param_dist = {"alpha": [0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 4, 8, 16, 32, 64, 128, 256],
              "fit_prior": [True, False]}

# run randomized search
n_iter_search = 36
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

# DTM
start = time()
random_search.fit(dtm_train, cuisine_label)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

# TF-IDF
start = time()
random_search.fit(tfidf_train, cuisine_label)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

#%%
clf = LinearSVC()
# specify parameters and distributions to sample from
param_dist = {"C": [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 4 ,8],
              "penalty": ["l2"],
              "dual":[True, False],
              "tol":[1e-2, 1e-3, 1e-4]}
# run randomized search
n_iter_search = 54
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

# DTM
start = time()
random_search.fit(dtm_train, cuisine_label)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

# TF-IDF
start = time()
random_search.fit(tfidf_train, cuisine_label)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

#%%
clf = LogisticRegression()
# specify parameters and distributions to sample from
param_dist = {"C": [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 4, 8],
              "penalty": [ "l2"],
              "dual":[True, False],
              "tol":[1e-2, 1e-3, 1e-4]}
# run randomized search
n_iter_search = 54
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

# DTM
start = time()
random_search.fit(dtm_train, cuisine_label)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

# TF-IDF
start = time()
random_search.fit(tfidf_train, cuisine_label)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)