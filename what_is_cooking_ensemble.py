# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 23:36:51 2015

@author: liangyupan
"""

from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.cross_validation import KFold, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import csv
import numpy as np
import pandas as pd
import re
import xgboost as xgb


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

# Document term matrix
vectorizer = CountVectorizer()
vectorizer.fit_transform(traindf['processed_ingredients_string'])
# print vectorizer.get_feature_names()
dtm_train = vectorizer.transform(traindf['processed_ingredients_string'])
# print strain.todense().shape
dtm_test = vectorizer.transform(testdf['processed_ingredients_string'])
# print stest.todense().shape


# Tf–idf term weighting
tfidf_trans = TfidfTransformer()
tfidf_train = tfidf_trans.fit_transform(dtm_train)
tfidf_test = tfidf_trans.fit_transform(dtm_test)


# Training classifiers
clf1 = RandomForestClassifier()
clf2 = AdaBoostClassifier()
clf3 = xgb.XGBClassifier()
clf4 = KNeighborsClassifier()
clf5 = DecisionTreeClassifier()

eclf = VotingClassifier(estimators=[('rf', clf1), ('ab', clf2), ('gb', clf3), ('ls', clf4), ('dt', clf5)], voting='soft', weights=[1, 0.5, 1.5, 1, 1])
eclf.fit(dtm_train, cuisine_label)
predict_result = eclf.predict(dtm_test)


testdf['cuisine'] = le.inverse_transform(predict_result)
predict_dict = dict(zip(testdf['id'], testdf['cuisine']) )
with open('predict_result_ensemble.csv', 'w') as csvfile:
    fieldnames = ['id', 'cuisine']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for key, value in predict_dict.iteritems():
        writer.writerow({'id': key, 'cuisine': value})
print 'finished'

#for clf, label in zip([clf1, clf2, clf3, eclf], ['Random Forest', 'Adaboost', 'Xgboost']):
#    scores = cross_validation.cross_val_score(clf, tfidf_train, cuisine_label, cv=2, scoring='accuracy')
#    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))