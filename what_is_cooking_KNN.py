# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import pandas as pd
import numpy as np
import xgboost as xgb
import re
import csv

#from sklearn import datasets #for the iris example
#from itertools import product
from sklearn import preprocessing
from operator import itemgetter
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import randint as sp_randint
#from sklearn.ensemble import VotingClassifier
from time import time
#import pickle
#from math import sqrt

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

'''
# Map cuisine to integer(start from 0)
item = set()
for i in traindf['cuisine']:
    item.add(i)
item = sorted(item)
cuisine_array = np.array(item)
cuisine_dict = dict(zip(cuisine_array, range(0, len(cuisine_array))))
traindf['cuisine_number'] = [cuisine_dict[n] for n in traindf['cuisine']]
cuisine_label = traindf['cuisine_number']   #our output in number
'''

# Encode cuisine to integer(start from 0)
le = preprocessing.LabelEncoder()
le.fit(traindf['cuisine'])
traindf['cuisine_number'] = le.transform(traindf['cuisine'])
cuisine_label = traindf['cuisine_number']

# Document term matrix
vectorizer = CountVectorizer()
vectorizer.fit_transform(traindf['processed_ingredients_string'])
dtm_train = vectorizer.transform(traindf['processed_ingredients_string']) #our input in numbers
dtm_test = vectorizer.transform(testdf['processed_ingredients_string']) #our testing data in numbers

# Tf–idf term
tfidf_trans = TfidfTransformer()
tfidf_train = tfidf_trans.fit_transform(dtm_train)
tfidf_test = tfidf_trans.fit_transform(dtm_test)

#Cross-validation test code; need to do random search first
'''
#from sklearn import neighbors

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
#knn = neighbors.KNeighborsClassifier(n_neighbors=20, leaf_size=100)

#these two lines will consume too much memory
#knn.fit(dtm_train, cuisine_label) 
#knn.score(dtm_train, cuisine_label)

#knn.fit(X_train, y_train)
#knn.score(X_train, y_train)
#predicted = knn.predict(X_test)

from sklearn import metrics
print metrics.classification_report(y_test, predicted)
print metrics.confusion_matrix(y_test, predicted)
print metrics.f1_score(y_test, predicted)   #could change another algorithm

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(knn, X, y, cv=5) #need to change?
print scores
'''

'''-------------------------------------------------------------'''
'''sampele randome forest with grid search and randomized search'''
'''-------------------------------------------------------------'''

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
# specify parameters and distributions to sample from
param_dist = {"n_neighbors": [10, 30, 50, 70],
              "weights": ["uniform", "distance"],
              "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
              "leaf_size": sp_randint(50, 300)}

# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)

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

print 'finished'