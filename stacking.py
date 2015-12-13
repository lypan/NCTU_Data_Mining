# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 01:04:42 2015

@author: liangyupan
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

#%%####################level 1
# cut train_data to train set and cv set
spilt_number = 20000;
n_folds = 5;


clf1 = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, n_estimators=275, min_samples_split=3, criterion='gini', max_features=43, max_depth=None)
# clf2 = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, n_estimators=150, min_samples_split=6, criterion='gini', max_features=40, max_depth=None)
clf3 = ExtraTreesClassifier(max_features=6, n_estimators=100, criterion='gini', max_depth=None)
# clf4 = ExtraTreesClassifier(max_features=1, n_estimators=70, criterion='entropy', max_depth=None)
clf5 = LinearSVC(penalty='l2', C=0.2, tol=0.0001, dual=False)
# clf6 = LinearSVC(penalty='l2', C=0.6, tol=0.01, dual=False)
clf7 = LogisticRegression(penalty='l2', C=1, tol=0.0001, dual=False)
# clf8 = LogisticRegression(penalty='l2', C=8, tol=0.01, dual=False)

num_rounds = 2000
param = {
   'objective':'multi:softmax',
   'eta':0.09,
   'max_depth':5,
   'num_class':20
}
clf9 = xgb.train(param, xgb.DMatrix(dtm_train[:spilt_number], cuisine_label[:spilt_number]), num_rounds)
# clf10 = xgb.train(param, xgb.DMatrix(tfidf_train[:spilt_number], cuisine_label[:spilt_number]), num_rounds)


#%%
# Ready for cross validation
skf = list(StratifiedKFold(cuisine_label, n_folds))


#rf on dtm
clf1.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred1 = clf1.predict(dtm_train[spilt_number:])
test_pred1 = clf1.predict(dtm_test)
#rf on tfidf
# clf2.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
# train_pred2 = clf2.predict(tfidf_train[spilt_number:])
# test_pred2 = clf2.predict(tfidf_test)
#ex on dtm
clf3.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred3 = clf3.predict(dtm_train[spilt_number:])
test_pred3 = clf3.predict(dtm_test)
#ex on tfidf
# clf4.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
# train_pred4 = clf4.predict(tfidf_train[spilt_number:])
# test_pred4 = clf4.predict(tfidf_test)
#%%
#lsvc on dtm
clf5.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred5 = clf5.predict(dtm_train[spilt_number:])
test_pred5 = clf5.predict(dtm_test)
#lsvc on tfidf
# clf6.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
# train_pred6 = clf6.predict(tfidf_train[spilt_number:])
# test_pred6 = clf6.predict(tfidf_test)
#lr on dtm
clf7.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred7 = clf7.predict(dtm_train[spilt_number:])
test_pred7 = clf7.predict(dtm_test)
#lr on tfidf
# clf8.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
# train_pred8 = clf8.predict(tfidf_train[spilt_number:])
# test_pred8 = clf8.predict(tfidf_test)
#xgb on dtm
train_pred9 = clf9.predict(xgb.DMatrix(dtm_train[spilt_number:]))
test_pred9 = clf9.predict(xgb.DMatrix(dtm_test))
#xgb on tfidf
# train_pred10 = clf10.predict(xgb.DMatrix(tfidf_train[spilt_number:]))
# test_pred10 = clf10.predict(xgb.DMatrix(tfidf_test))


#%%
blend_train = np.column_stack((train_pred1, train_pred3, train_pred5, train_pred7, train_pred9))
blend_test = np.column_stack((test_pred1, test_pred3, test_pred5, test_pred7, test_pred9))

#%%####################level 2
param = {
   'objective':'multi:softmax',
   'eta':0.09,
   'max_depth':5,
   'num_class':20
}
num_rounds = 2000


dtrain = xgb.DMatrix(blend_train, cuisine_label[spilt_number:])
dtest = xgb.DMatrix(blend_test)

xgbclf = xgb.train(param, dtrain, num_rounds)
predict_result = xgbclf.predict(dtest)
#%%####################write csv
testdf['cuisine'] = le.inverse_transform(predict_result.astype(np.int32))
predict_dict = dict(zip(testdf['id'], testdf['cuisine']))
with open('predict_result_ensemble.csv', 'w') as csvfile:
    fieldnames = ['id', 'cuisine']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for key, value in predict_dict.iteritems():
        writer.writerow({'id': key, 'cuisine': value})

print("--- %s seconds ---" % (time.time() - start_time))

