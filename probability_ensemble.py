# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 04:54:45 2015
@author: lypan
"""

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
from sklearn.linear_model import SGDClassifier
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


#%%
#tf-idf transforming
tfidf_trans = TfidfTransformer()
tfidf_train = tfidf_trans.fit_transform(dtm_train).toarray()
tfidf_test = tfidf_trans.transform(dtm_test).toarray()

#%%
clf1 = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, n_estimators=275, min_samples_split=3, criterion='gini', max_features=43, max_depth=None)
clf2 = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, n_estimators=150, min_samples_split=6, criterion='gini', max_features=40, max_depth=None)
clf3 = ExtraTreesClassifier(max_features=6, n_estimators=100, criterion='gini', max_depth=None)
clf4 = ExtraTreesClassifier(max_features=1, n_estimators=70, criterion='entropy', max_depth=None)
clf5 = SGDClassifier(penalty='l2', loss='log')
clf6 = SGDClassifier(penalty='l2', loss='modified_huber')
clf7 = LogisticRegression(penalty='l2', C=1, tol=0.0001, dual=False)
clf8 = LogisticRegression(penalty='l2', C=8, tol=0.01, dual=False)

num_rounds = 2000
param = {
   'objective':'multi:softprob',
   'eta':0.09,
   'max_depth':5,
   'num_class':20
}
clf9 = xgb.train(param, xgb.DMatrix(dtm_train, cuisine_label), num_rounds)
clf10 = xgb.train(param, xgb.DMatrix(tfidf_train, cuisine_label), num_rounds)


#%%
#rf on dtm
clf1.fit(dtm_train, cuisine_label)
test_pred1 = clf1.predict_proba(dtm_test).reshape(9944, 20)
#rf on tfidf
clf2.fit(tfidf_train, cuisine_label)
test_pred2 = clf2.predict_proba(tfidf_test).reshape(9944, 20)
#ex on dtm
clf3.fit(dtm_train, cuisine_label)
test_pred3 = clf3.predict_proba(dtm_test).reshape(9944, 20)
#ex on tfidf
clf4.fit(tfidf_train, cuisine_label)
test_pred4 = clf4.predict_proba(tfidf_test).reshape(9944, 20)
#%%
#lsvc on dtm
clf5.fit(dtm_train, cuisine_label)
test_pred5 = clf5.predict_proba(dtm_test).reshape(9944, 20)
#lsvc on tfidf
clf6.fit(tfidf_train, cuisine_label)
test_pred6 = clf6.predict_proba(tfidf_test).reshape(9944, 20)
#lr on dtm
clf7.fit(dtm_train, cuisine_label)
test_pred7 = clf7.predict_proba(dtm_test).reshape(9944, 20)
#lr on tfidf
clf8.fit(tfidf_train, cuisine_label)
test_pred8 = clf8.predict_proba(tfidf_test).reshape(9944, 20)
#xgb on dtm
test_pred9 = clf9.predict(xgb.DMatrix(dtm_test)).reshape(9944, 20)
#xgb on tfidf
test_pred10 = clf10.predict(xgb.DMatrix(tfidf_test)).reshape(9944, 20)


#%%
blend_prob = test_pred1 + test_pred2 + test_pred3 + test_pred4 + test_pred5 + test_pred6 + test_pred7 + test_pred8 + test_pred9 + test_pred10 
predict_result = np.argmax(blend_prob, axis=1) 
#%%
testdf['cuisine'] = le.inverse_transform(predict_result.astype(np.int32))
predict_dict = dict(zip(testdf['id'], testdf['cuisine']) )
with open('predict_result_ensemble.csv', 'w') as csvfile:
   fieldnames = ['id', 'cuisine']
   writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

   writer.writeheader()
   for key, value in predict_dict.iteritems():
       writer.writerow({'id': key, 'cuisine': value})

print("--- %s seconds ---" % (time.time() - start_time))
