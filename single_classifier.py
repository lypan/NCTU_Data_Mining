# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 22:57:27 2015

@author: liangyupan
"""

#%%
import re
import csv
import random
import time
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
#%%#################### original data
# read train and test data from json
start_time = time.time()
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
dtm_train = vectorizer.fit_transform(traindf['processed_ingredients_string'])
dtm_test = vectorizer.transform(testdf['processed_ingredients_string'])


#tf-idf transforming
tfidf_trans = TfidfTransformer()
tfidf_train = tfidf_trans.fit_transform(dtm_train)
tfidf_test = tfidf_trans.transform(dtm_test)

## 0-1 standardization
#std_trans = StandardScaler()
#std_train = std_trans.fit_transform(dtm_train)
#std_test = std_trans.transform(dtm_test)
#
## PCA
#pca = PCA(n_components=1000)
#pca_train = pca.fit(dtm_train).transform(dtm_train)
#pca_test = pca.transform(dtm_test)
#%%#################### xgboost parameter testing
param = {
   'objective':'multi:softprob',
   'eta':0.09,
   'max_depth':5,
   'num_class':20
}
num_rounds = 2000
clf = xgb.train(param, xgb.DMatrix(tfidf_train, cuisine_label), num_rounds)
#%%#################### xgboost cv
#xgb.cv(param, xgb.DMatrix(tfidf_train, cuisine_label), num_rounds, nfold=5,
#       metrics={'merror'}, seed = 0)
predict_result = clf.predict(xgb.DMatrix(tfidf_test)).reshape(9944, 20)
#%%####################write csv
#testdf['cuisine'] = le.inverse_transform(predict_result.astype(np.int32))
#predict_dict = dict(zip(testdf['id'], testdf['cuisine']) )
#with open('predict_result_ensemble.csv', 'w') as csvfile:
#    fieldnames = ['id', 'cuisine']
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#    writer.writeheader()
#    for key, value in predict_dict.iteritems():
#        writer.writerow({'id': key, 'cuisine': value})

print("--- %s seconds ---" % (time.time() - start_time))