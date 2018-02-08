
# coding: utf-8

# In[1]:

import pandas as pd
import string
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[2]:

train_all = pd.read_table('train.tsv')
test_all = pd.read_table('test.tsv',index_col=0)


# In[11]:

train_all.head()


# In[3]:

train_all['comment'] = train_all['comment'].apply(lambda x:''.join([i.lower() for i in x 
                                                  if i not in string.punctuation])).str.replace('\d+', '')
test_all['comment'] = test_all['comment'].apply(lambda x:''.join([i.lower() for i in x 
                                                  if i not in string.punctuation])).str.replace('\d+', '')


# In[4]:

train_all.head()


# In[30]:

test_all.head()


# In[5]:

from sklearn.model_selection import train_test_split

train, test = train_test_split(train_all, test_size = 0.2)
train.head()


# In[39]:

train, test = train_test_split(train_all, test_size = 0.2)

pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),max_df = 0.9,stop_words='english')),
                  ('clf', RandomForestClassifier(n_estimators=12))])
pipeline.fit(train.comment, train.label)

predicted = pipeline.predict(test.comment)
np.mean(predicted == test.label)


# In[37]:

from sklearn.svm import LinearSVC

train, test = train_test_split(train_all, test_size = 0.2)

pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,3),max_df = 0.92,stop_words='english')),
                  ('clf', LinearSVC(C=0.96))])
pipeline.fit(train_all.comment, train_all.label)

predicted = pipeline.predict(test_all.comment)
# np.mean(predicted == test.label)


# In[47]:

from sklearn.ensemble import ExtraTreesClassifier

train, test = train_test_split(train_all, test_size = 0.2)

pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),max_df = 0.95,stop_words='english')),
                  ('clf', ExtraTreesClassifier(n_estimators =12))])
pipeline.fit(train.comment, train.label)

predicted = pipeline.predict(test.comment)
np.mean(predicted == test.label)


# In[34]:

from sklearn.svm import LinearSVC

train, test = train_test_split(train_all, test_size = 0.2)

c_value = [0.95,0.96,0.97,0.98,0.98,0.99,1,1.01,1.02,1.03]
y_value = []
for k in c_value:
    print("k=",k)
    pipeline = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,3),max_df = 0.92,stop_words='english')),
                      ('clf', LinearSVC(C=k))])
    pipeline.fit(train.comment, train.label)

    predicted = pipeline.predict(test.comment)
    y_value.append(np.mean(predicted == test.label))


# In[36]:

sorted(list(zip(c_value,y_value)),key=lambda x:x[1])


# In[35]:

import matplotlib.pyplot as plt

plt.plot(np.array(c_value),np.array(y_value))
plt.show()


# In[38]:

result = pd.DataFrame(predicted)

result.columns = ['Category']
result.to_csv('result202.csv',index_label='Id')


# In[41]:

result['comment']=test_all.comment
result[result.Category==1].head()

