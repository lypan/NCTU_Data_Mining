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
import xgboost as xgb


#%%#################### original data
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

## 0-1 standardization
#std_trans = StandardScaler()
#std_train = std_trans.fit_transform(dtm_train)
#std_test = std_trans.transform(dtm_test)
#
## PCA
#pca = PCA(n_components=1000)
#pca_train = pca.fit(dtm_train).transform(dtm_train)
#pca_test = pca.transform(dtm_test)


#%%####################level 1
# cut train_data to train set and cv set
spilt_number = 20000;
n_folds = 5;


clf1 = RandomForestClassifier(bootstrap=False, min_samples_leaf=1, n_estimators=275, min_samples_split=3, criterion='gini', max_features=43, max_depth=None)
# clf2 = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
clf3 = ExtraTreesClassifier(max_features=6, n_estimators=100, criterion='gini', max_depth=None)
# clf4 = ExtraTreesClassifier(n_estimators = 100 * 2, criterion = 'gini')
clf5 = KNeighborsClassifier(n_neighbors=16, weights='distance', leaf_size=178, algorithm='ball_tree')
# clf6 = KNeighborsClassifier(n_neighbors = 380, metric = 'cosine', algorithm = 'brute')
clf7 = MultinomialNB(alpha=0.4, fit_prior=True)
# clf8 = MultinomialNB()
num_rounds = 200
param = {
   'objective':'multi:softmax',
   'eta':0.3,
   'max_depth':24,
   'num_class':20,
   'colsample_bytree':0.3,
   'min_child_weight':1
}
clf9 = xgb.train(param, xgb.DMatrix(dtm_train[:spilt_number], cuisine_label[:spilt_number]), num_rounds)
# clf10 = xgb.train(param, xgb.DMatrix(tfidf_train[:spilt_number], cuisine_label[:spilt_number]), num_rounds)
clf11 = LinearSVC(penalty='l2', C=0.2, dual=False)
# clf12 = LinearSVC()
clf13 = LogisticRegression(penalty='l2', C=1, dual=False)
# clf14 = LogisticRegression()




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
#knn on dtm
clf5.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred5 = clf5.predict(dtm_train[spilt_number:])
test_pred5 = clf5.predict(dtm_test)
# #knn on tfidf
# clf6.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
# train_pred6 = clf6.predict(tfidf_train[spilt_number:])
# test_pred6 = clf6.predict(tfidf_test)
#mb on dtm
clf7.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred7 = clf7.predict(dtm_train[spilt_number:])
test_pred7 = clf7.predict(dtm_test)
#mb on tfidf
# clf8.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
# train_pred8 = clf8.predict(tfidf_train[spilt_number:])
# test_pred8 = clf8.predict(tfidf_test)
#xgb on dtm
train_pred9 = clf9.predict(xgb.DMatrix(dtm_train[spilt_number:]))
test_pred9 = clf9.predict(xgb.DMatrix(dtm_test))
#xgb on tfidf
# train_pred10 = clf10.predict(xgb.DMatrix(tfidf_train[spilt_number:]))
# test_pred10 = clf10.predict(xgb.DMatrix(tfidf_test))
clf11.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred11 = clf11.predict(dtm_train[spilt_number:])
test_pred11 = clf11.predict(dtm_test)

clf13.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred13 = clf13.predict(dtm_train[spilt_number:])
test_pred13 = clf13.predict(dtm_test)
#%%
# blend_train = np.column_stack((train_pred1, train_pred2, train_pred3, train_pred4, train_pred5, train_pred6, train_pred7, train_pred8, train_pred9, train_pred10))
# blend_test = np.column_stack((test_pred1, test_pred2, test_pred3, test_pred4, test_pred5, test_pred6, test_pred7, test_pred8, test_pred9, test_pred10))

blend_train = np.column_stack((train_pred1, train_pred3, train_pred5, train_pred7, train_pred9, train_pred11, train_pred13))
blend_test = np.column_stack((test_pred1, test_pred3, test_pred5, test_pred7, test_pred9, test_pred11, test_pred13))

#%%####################level 2
param = {
   'objective':'multi:softmax',
   'eta':0.3,
   'max_depth':24,
   'num_class':20,
   'colsample_bytree':0.3,
   'min_child_weight':1
}
evals_result = {}
num_rounds = 200

l2_row_num = blend_train.shape[0]
l2_row_spl = int(l2_row_num * 0.8)


dtrain = xgb.DMatrix(blend_train[:l2_row_spl], cuisine_label[spilt_number:spilt_number + l2_row_spl])
deval = xgb.DMatrix(blend_train[l2_row_spl:], cuisine_label[spilt_number + l2_row_spl:])
dtest = xgb.DMatrix(blend_test)
watchlist = [(dtrain,'train'), (deval,'eval')]

xgbclf = xgb.train(param, dtrain, num_rounds, watchlist, early_stopping_rounds=50, evals_result=evals_result)
predict_result = xgbclf.predict(dtest, ntree_limit=xgbclf.best_iteration)
#%%####################write csv
testdf['cuisine'] = le.inverse_transform(predict_result.astype(np.int32))
predict_dict = dict(zip(testdf['id'], testdf['cuisine']) )
with open('predict_result_ensemble.csv', 'w') as csvfile:
    fieldnames = ['id', 'cuisine']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for key, value in predict_dict.iteritems():
        writer.writerow({'id': key, 'cuisine': value})
