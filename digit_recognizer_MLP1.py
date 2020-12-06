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


# In[13]:


# データを正規化 (255 で割る)
train_data_x = train_data_x / 255
test_data = test_data / 255
train_data_x.describe()


# In[14]:


# MLP で学習
# https://qiita.com/maskot1977/items/d0253e1eab1ff1315dff
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

train_size = 500
test_size = 100

data_train, data_test, label_train, label_test = train_test_split(train_data_x, train_data_y, test_size=test_size, train_size=train_size, random_state=1)

clf = MLPClassifier(max_iter=10000)
clf.fit(data_train, label_train)
print(clf.score(data_train, label_train))


# In[15]:


plt.title("Loss Curve")
plt.plot(clf.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()


# In[18]:


from sklearn import metrics
prediction = clf.predict(data_train)
co_mat = metrics.confusion_matrix(label_train, prediction)
print(co_mat)


# In[19]:


clf = MLPClassifier(max_iter=10000)
clf.fit(train_data_x, train_data_y)
print(clf.score(train_data_x, train_data_y))


# In[20]:


plt.title("Loss Curve")
plt.plot(clf.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()


# In[21]:


prediction = clf.predict(train_data_x)
co_mat = metrics.confusion_matrix(train_data_y, prediction)
print(co_mat)


# In[22]:


prediction = clf.predict(test_data)
output = pd.DataFrame({"ImageId" : np.arange(1, 28000+1), "Label":prediction})
output.head()


# In[23]:


output.to_csv('digit_recognizer_MLP1.csv', index=False)
print("Your submission was successfully saved!")

