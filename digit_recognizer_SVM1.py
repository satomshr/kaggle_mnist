#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

# データ確認
train_data.info()


# In[2]:


# データ確認
train_data.describe()


# In[3]:


# データ確認
test_data.info()


# In[4]:


# データ確認
test_data.describe()


# In[5]:


train_data.head()


# In[6]:


test_data.head()


# In[7]:


# label を抽出
train_data_y = train_data["label"]


# In[8]:


# データを抽出
train_data_x = train_data.drop(columns="label")


# In[9]:


train_data_x.head()


# In[10]:


train_data_y.head()


# In[11]:


# データを正規化 (255 で割る)
train_data_x = train_data_x / 255
test_data = test_data / 255
train_data_x.describe()


# In[12]:


# SVM で学習

from sklearn import svm
from sklearn import metrics

clf = svm.SVC()
clf.fit(train_data_x, train_data_y)
prediction = clf.predict(train_data_x)
accuracy_score = metrics.accuracy_score(train_data_y, prediction)
print(accuracy_score)


# In[13]:


co_mat = metrics.confusion_matrix(train_data_y, prediction)
print(co_mat)


# In[14]:


prediction = clf.predict(test_data)
output = pd.DataFrame({"ImageId" : np.arange(1, 28000+1), "Label":prediction})
output.head()


# In[ ]:





# In[ ]:





# In[15]:


output.to_csv('digit_recognizer_SVM1.csv', index=False)
print("Your submission was successfully saved!")

