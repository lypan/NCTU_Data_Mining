# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold, train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np
import xgboost as xgb
import re
import csv


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


# Document term matrix
vectorizer = CountVectorizer()
vectorizer.fit_transform(traindf['processed_ingredients_string'])
# print vectorizer.get_feature_names()
strain = vectorizer.transform(traindf['processed_ingredients_string'])
# print strain.todense().shape
stest = vectorizer.transform(testdf['processed_ingredients_string'])
# print stest.todense().shape


# Map cuisine to integer(start from 0)
item = set()
for i in traindf['cuisine']:
    item.add(i)
item = sorted(item)

cuisine_array = np.array(item)
cuisine_dict = dict(zip(cuisine_array, range(0, len(cuisine_array))))

traindf['cuisine_number'] = [cuisine_dict[n] for n in traindf['cuisine']]
cuisine_label = traindf['cuisine_number']

# xgboost need its own data structure Dmatrix
dtrain = xgb.DMatrix(strain, label = cuisine_label, missing = 0)
dtest = xgb.DMatrix(stest)


# set xgboost parameter
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.3
param['max_depth'] = 25
param['nthread'] = 8
param['num_class'] = 20
num_round = 200
# watchlist = [ (xg_train,'train'), (xg_test, 'test') ]


# start prediction
bst = xgb.train(param, dtrain, num_round)
pred = bst.predict(dtest)
#xgb.plot_importance(bst)


xgb.cv(param, dtrain, num_round, nfold=5,
       metrics={'merror'}, seed = 0)

# rng = np.random.RandomState(31337)
# kf = KFold(traindf.shape[0], n_folds=2, shuffle=True, random_state=rng)
# for train_index, test_index in kf:
#     xgb_model = xgb.XGBClassifier().fit(strain.todense()[train_index], traindf['cuisine_number'][train_index])
#     predictions = xgb_model.predict(strain.todense()[test_index])
#     actuals = traindf['cuisine_number'][test_index]
#     print(confusion_matrix(actuals, predictions))


testdf['cuisine'] = [cuisine_array[int(n)] for n in pred]
predict_dict = dict(zip(testdf['id'], testdf['cuisine']) )
with open('predict_result.csv', 'w') as csvfile:
    fieldnames = ['id', 'cuisine']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for key, value in predict_dict.iteritems():
        writer.writerow({'id': key, 'cuisine': value})
        

print 'finished'