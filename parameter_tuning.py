# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:00:35 2015

@author: liangyupan
"""

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.stats import randint as sp_randint
from sklearn.ensemble import VotingClassifier
from time import time
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import re
import csv
from math import sqrt

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


print 'started'
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


# Tf–idf term
tfidf_trans = TfidfVectorizer(stop_words='english',
                             ngram_range = ( 1 , 1 ),analyzer="word",
                             max_df = .57 , binary=False , token_pattern=r'\w+' , sublinear_tf=False)

#tfidf_trans = TfidfTransformer()
tfidf_train = tfidf_trans.fit_transform(traindf['processed_ingredients_string'])
tfidf_test = tfidf_trans.fit_transform(testdf['processed_ingredients_string'])


clf = RandomForestClassifier()
# specify parameters and distributions to sample from
param_dist = {"n_estimators": [50, 75, 100],
              "max_depth": [3, None],
              "max_features": sp_randint(57, 69),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(strain, cuisine_label)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)



clf = RandomForestClassifier(n_estimators=100)
label = "RF"
scores = cross_validation.cross_val_score(clf, strain, cuisine_label, cv=2, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

scores = cross_validation.cross_val_score(clf, tfidf_train, cuisine_label, cv=2, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# Document term matrix
vectorizer = CountVectorizer()
vectorizer.fit_transform(traindf['processed_ingredients_string'])
# print vectorizer.get_feature_names()
strain = vectorizer.transform(traindf['processed_ingredients_string'])
# print strain.todense().shape
stest = vectorizer.transform(testdf['processed_ingredients_string'])
# print stest.todense().shape
kf = KFold(n=len(traindf), n_folds=5)

#Random forest
scores = []
for train_index, test_index in kf:
    X_train, X_test = strain[train_index], strain[test_index]
    y_train, y_test = cuisine_label[train_index], cuisine_label[test_index]
    alg = RandomForestClassifier(n_estimators=10)
    alg.fit(X_train, y_train)
    scores.append(alg.score(X_test, y_test))
print scores
scores = []
for train_index, test_index in kf:
    X_train, X_test = tfidf_train[train_index], tfidf_train[test_index]
    y_train, y_test = cuisine_label[train_index], cuisine_label[test_index]
    alg = RandomForestClassifier(n_estimators=10)
    alg.fit(X_train, y_train)
    scores.append(alg.score(X_test, y_test))
print scores

#SVM
from sklearn.svm import LinearSVC
scores = []
for train_index, test_index in kf:
    X_train, X_test = strain[train_index], strain[test_index]
    y_train, y_test = cuisine_label[train_index], cuisine_label[test_index]
    alg = LinearSVC()
    alg.fit(X_train, y_train)
    scores.append(alg.score(X_test, y_test))