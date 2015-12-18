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
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
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

#tf-idf transforming
tfidf_trans = TfidfTransformer()
tfidf_train = tfidf_trans.fit_transform(dtm_train).toarray()
tfidf_test = tfidf_trans.transform(dtm_test).toarray()

#%%
## Mean validation score: 0.764 (std: 0.002)
#clf1 = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, n_estimators=275, min_samples_split=3, criterion='gini', max_features=43, max_depth=None)
## Mean validation score: 0.759 (std: 0.002)
#clf2 = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, n_estimators=150, min_samples_split=6, criterion='gini', max_features=40, max_depth=None)
## Mean validation score: 0.765 (std: 0.003)
#clf3 = ExtraTreesClassifier(max_features=6, n_estimators=100, criterion='gini', max_depth=None)
## Mean validation score: 0.748 (std: 0.002)
#clf4 = ExtraTreesClassifier(max_features=1, n_estimators=70, criterion='entropy', max_depth=None)
## Mean validation score: 0.763 (std: 0.006)
#clf5 = SGDClassifier(penalty='l2', loss='log')
## Mean validation score: 0.782 (std: 0.001)
#clf6 = SGDClassifier(penalty='l2', loss='modified_huber')
## Mean validation score: 0.781 (std: 0.000)
#clf7 = LogisticRegression(penalty='l2', C=1, tol=0.0001, dual=False)
## Mean validation score: 0.786 (std: 0.001)
#clf8 = LogisticRegression(penalty='l2', C=8, tol=0.01, dual=False)
#
#num_rounds = 2000
#param = {
#   'objective':'multi:softprob',
#   'eta':0.09,
#   'max_depth':5,
#   'num_class':20
#}
## Mean validation score: 0.8 
#clf9 = xgb.train(param, xgb.DMatrix(dtm_train, cuisine_label), num_rounds)
## Mean validation score: 0.8 
#clf10 = xgb.train(param, xgb.DMatrix(tfidf_train, cuisine_label), num_rounds)
## Mean validation score: 0.648 (std: 0.003)
#clf11 = KNeighborsClassifier(n_neighbors=16, weights='distance', algorithm='ball_tree', leaf_size=178)
## Mean validation score: 0.747 (std: 0.002)
#clf12 = KNeighborsClassifier(n_neighbors=16, weights='distance', algorithm='auto', leaf_size=105)
## Mean validation score: 0.726 (std: 0.001)
#clf13 = MultinomialNB(alpha=0.4, fit_prior=True)
## Mean validation score: 0.734 (std: 0.001)
#clf14 = MultinomialNB(alpha=0.05, fit_prior=True)

#%%
##rf on dtm
#clf1.fit(dtm_train, cuisine_label)
##rf on tfidf
#clf2.fit(tfidf_train, cuisine_label)
##ex on dtm
#clf3.fit(dtm_train, cuisine_label)
##ex on tfidf
#clf4.fit(tfidf_train, cuisine_label)
##lsvc on dtm
#clf5.fit(dtm_train, cuisine_label)
##lsvc on tfidf
#clf6.fit(tfidf_train, cuisine_label)
##lr on dtm
#clf7.fit(dtm_train, cuisine_label)
##lr on tfidf
#clf8.fit(tfidf_train, cuisine_label)
##xgb on dtm
###(already fit)
##xgb on tfidf
##(already fit)
###knn on dtm
#clf11.fit(dtm_train, cuisine_label)
##knn on tfidf
#clf12.fit(tfidf_train, cuisine_label)
##multinb on dtm
#clf13.fit(dtm_train, cuisine_label)
##multinb on tfidf
#clf14.fit(tfidf_train, cuisine_label)

#%% save model
#joblib.dump(clf1, 'RandomForestClassifier_raw.pkl') 
#joblib.dump(clf2, 'RandomForestClassifier_tfidf.pkl') 
#joblib.dump(clf3, 'ExtraTreesClassifier_raw.pkl') 
#joblib.dump(clf4, 'ExtraTreesClassifier_tfidf.pkl') 
#joblib.dump(clf5, 'SGDClassifier_raw.pkl') 
#joblib.dump(clf6, 'SGDClassifier_tfidf.pkl') 
#joblib.dump(clf7, 'LogisticRegression_raw.pkl') 
#joblib.dump(clf8, 'LogisticRegression_tfidf.pkl') 
#clf9.save_model('xgboost_raw.model')
#clf10.save_model('xgboost_tfidf.model')
#joblib.dump(clf11, 'KNeighborsClassifier_raw.pkl') 
#joblib.dump(clf12, 'KNeighborsClassifier_tfidf.pkl') 
#joblib.dump(clf13, 'MultinomialNB_raw.pkl') 
#joblib.dump(clf14, 'MultinomialNB_tfidf.pkl') 

#%% predict result
#test_pred1 = clf1.predict_proba(dtm_test)
#test_pred2 = clf2.predict_proba(tfidf_test)
#test_pred3 = clf3.predict_proba(dtm_test)
#test_pred4 = clf4.predict_proba(tfidf_test)
#test_pred5 = clf5.predict_proba(dtm_test)
#test_pred6 = clf6.predict_proba(tfidf_test)
#test_pred7 = clf7.predict_proba(dtm_test)
#test_pred8 = clf8.predict_proba(tfidf_test)
#test_pred9 = clf9.predict(xgb.DMatrix(dtm_test))
#test_pred10 = clf10.predict(xgb.DMatrix(tfidf_test))
#test_pred11 = clf11.predict_proba(dtm_test)
#test_pred12 = clf12.predict_proba(tfidf_test)
#test_pred13 = clf13.predict_proba(dtm_test)
#test_pred14 = clf14.predict_proba(tfidf_test)

#%% save result
#np.save("RandomForestClassifier_raw_predict", test_pred1)
#np.save("RandomForestClassifier_tfidf_predict", test_pred2)
#np.save("ExtraTreesClassifier_raw_predict", test_pred3)
#np.save("ExtraTreesClassifier_tfidf_predict", test_pred4)
#np.save("SGDClassifier_raw_predict", test_pred5)
#np.save("SGDClassifier_tfidf_predict", test_pred6)
#np.save("LogisticRegression_raw_predict", test_pred7)
#np.save("LogisticRegression_tfidf_predict", test_pred8)
#np.save("xgboost_raw_predict", test_pred9)
#np.save("xgboost_tfidf_predict", test_pred10)
#np.save("KNeighborsClassifier_raw_predict", test_pred11)
#np.save("KNeighborsClassifier_tfidf_predict", test_pred12)
#np.save("MultinomialNB_raw_predict", test_pred13)
#np.save("MultinomialNB_tfidf_predict", test_pred14)
#%% load model
clf1 = joblib.load('RandomForestClassifier_raw.pkl')
clf2 = joblib.load('RandomForestClassifier_tfidf.pkl') 
clf3 = joblib.load('ExtraTreesClassifier_raw.pkl')
clf4 = joblib.load('ExtraTreesClassifier_tfidf.pkl') 
clf5 = joblib.load('SGDClassifier_raw.pkl')
clf6 = joblib.load('SGDClassifier_tfidf.pkl') 
clf7 = joblib.load('LogisticRegression_raw.pkl')
clf8 = joblib.load('LogisticRegression_tfidf.pkl') 
clf9 = xgb.Booster({'nthread':8}) 
clf9.load_model("xgboost_raw.model") 
clf10 = xgb.Booster({'nthread':8}) 
clf10.load_model("xgboost_tfidf.model") 
clf11 = joblib.load('KNeighborsClassifier_raw.pkl')
clf12 = joblib.load('KNeighborsClassifier_tfidf.pkl') 
clf13 = joblib.load('MultinomialNB_raw.pkl')
clf14 = joblib.load('MultinomialNB_tfidf.pkl') 
#%% load predict
test_pred1 = np.load("RandomForestClassifier_raw_predict.npy")
test_pred2 = np.load("RandomForestClassifier_tfidf_predict.npy")
test_pred3 = np.load("ExtraTreesClassifier_raw_predict.npy")
test_pred4 = np.load("ExtraTreesClassifier_tfidf_predict.npy")
test_pred5 = np.load("SGDClassifier_raw_predict.npy")
test_pred6 = np.load("SGDClassifier_tfidf_predict.npy")
test_pred7 = np.load("LogisticRegression_raw_predict.npy")
test_pred8 = np.load("LogisticRegression_tfidf_predict.npy")
test_pred9 = np.load("xgboost_raw_predict.npy")
test_pred10 = np.load("xgboost_tfidf_predict.npy")
test_pred11 = np.load("KNeighborsClassifier_raw_predict.npy")
test_pred12 = np.load("KNeighborsClassifier_tfidf_predict.npy")
test_pred13 = np.load("MultinomialNB_raw_predict.npy")
test_pred14 = np.load("MultinomialNB_tfidf_predict.npy")
blend_predict = np.column_stack((test_pred1.flatten(), test_pred2.flatten(), test_pred3.flatten(), test_pred4.flatten(), test_pred5.flatten(), test_pred6.flatten(), test_pred7.flatten(), test_pred8.flatten(), test_pred9.flatten(), test_pred10.flatten(), test_pred11.flatten(), test_pred12.flatten(), test_pred13.flatten(), test_pred14.flatten()))
#%% calculate covariance matrix
#coefficient_matrix = np.zeros((14, 14))
#for i in range(14):
#    for j in range(14):
#        coefficient_matrix[i][j] = np.corrcoef(blend_predict[:, i],blend_predict[:, j])[0][1]
#np.savetxt('correlation_matrix.txt', coefficient_matrix, fmt='%.3f')
#%%
blend_prob = test_pred1 + test_pred2 + test_pred3 + test_pred4 + test_pred5 + test_pred6 + test_pred7 + test_pred8 + test_pred9 + test_pred10 + test_pred11 + test_pred12 + test_pred13 + test_pred14
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
