# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 01:04:42 2015

@author: liangyupan
"""
#%%
import random
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

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

random.shuffle(dtm_train)

# tf-idf transforming
tfidf_trans = TfidfTransformer()
tdif_train = tfidf_trans.fit_transform(dtm_train).toarray()
tdif_test = tfidf_trans.transform(dtm_test).toarray()

# 0-1 standardization
std_trans = StandardScaler()
std_train = std_trans.fit_transform(dtm_train)
std_test = std_trans.transform(dtm_test)

# PCA
pca = PCA(n_components=1000)
pca_train = pca.fit(dtm_train).transform(dtm_train)
pca_test = pca.transform(dtm_test)


#%%####################level 1
# cut train_data to train set and cv set
spilt_number = 20000;
n_folds = 5;

clf1 = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
clf2 = RandomForestClassifier(n_estimators = 100, criterion = 'gini')
clf3 = ExtraTreesClassifier(n_estimators = 100 * 2, criterion = 'gini')
clf4 = ExtraTreesClassifier(n_estimators = 100 * 2, criterion = 'gini')
clf5 = GradientBoostingClassifier(n_estimators = 100)
clf6 = GradientBoostingClassifier(n_estimators = 100)
clf7 = MultinomialNB()
clf8 = MultinomialNB()



#%%
# Ready for cross validation
skf = list(StratifiedKFold(cuisine_label, n_folds))


#rf on raw
clf1.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred1 = clf1.predict(dtm_train[spilt_number:])
test_pred1 = clf1.predict(dtm_test)
#rf on tfidf
clf2.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
train_pred2 = clf2.predict(tfidf_train[spilt_number:])
test_pred2 = clf2.predict(dtm_test)
#ex on raw
clf3.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred3 = clf3.predict(dtm_train[spilt_number:])
test_pred3 = clf3.predict(dtm_test)
#ex on tfidf
clf4.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
train_pred4 = clf4.predict(tfidf_train[spilt_number:])
test_pred4 = clf4.predict(dtm_test)
#gb on raw
clf5.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred5 = clf5.predict(dtm_train[spilt_number:])
test_pred5 = clf5.predict(dtm_test)
#gb on tfidf
clf6.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
train_pred6 = clf6.predict(tfidf_train[spilt_number:])
test_pred6 = clf6.predict(dtm_test)
#mb on raw
clf7.fit(dtm_train[:spilt_number], cuisine_label[:spilt_number])
train_pred7 = clf7.predict(dtm_train[spilt_number:])
test_pred7 = clf7.predict(dtm_test)
#mb on tfidf
clf8.fit(tfidf_train[:spilt_number], cuisine_label[:spilt_number])
train_pred8 = clf8.predict(tfidf_train[spilt_number:])
test_pred8 = clf8.predict(dtm_test)

blend_train = np.hstack((train_pred1, train_pred2, train_pred3, train_pred4, train_pred5, train_pred6, train_pred7, train_pred8))
blend_test = np.hstack((test_pred1, test_pred2, test_pred3, test_pred4, test_pred5, test_pred6, test_pred7, test_pred8))
#%%####################level 2
bclf = LogisticRegression()
bclf.fit(blend_train, cuisine_label[spilt_number:])
predict_result = bclf.predict(blend_test)


#%%####################write csv
testdf['cuisine'] = le.inverse_transform(predict_result)
predict_dict = dict(zip(testdf['id'], testdf['cuisine']) )
with open('predict_result_ensemble.csv', 'w') as csvfile:
    fieldnames = ['id', 'cuisine']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for key, value in predict_dict.iteritems():
        writer.writerow({'id': key, 'cuisine': value})
