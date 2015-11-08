
# coding: utf-8

# In[33]:


from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.linear_model import LogisticRegression


# A combination of Word lemmatization + LinearSVC model finally pushes the accuracy score past 80%

traindf = pd.read_json("train.json")
traindf['ingredients_clean_string'] = [' , '.join(z).strip() for z in traindf['ingredients']]  
traindf['ingredients_string'] = [' '.join([WordNetLemmatizer().lemmatize(re.sub('[^A-Za-z]', ' ', line)) for line in lists]).strip() for lists in traindf['ingredients']]       


# In[34]:

ingredient = traindf.loc[:, 'ingredients_clean_string']
item = set()


# In[36]:

for i in ingredient:
    for j in i.split(','):
        j = j.encode('ascii', 'ignore')
        j = j.strip()
        item.add(j)


# In[38]:

item = sorted(item)


# In[39]:

print item

