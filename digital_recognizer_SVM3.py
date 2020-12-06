#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

# データ確認
train_data.info()


# In[39]:


# データ確認
train_data.describe()


# In[40]:


# データ確認
test_data.info()


# In[41]:


# データ確認
test_data.describe()


# In[42]:


train_data.head()


# In[43]:


test_data.head()


# In[44]:


# label を抽出
train_data_y = train_data["label"]


# In[45]:


# データを抽出
train_data_x = train_data.drop(columns="label")


# In[46]:


train_data_x.head()


# In[47]:


train_data_y.head()


# In[48]:


# データを正規化 (255 で割る)
train_data_x = train_data_x / 255
test_data = test_data / 255
train_data_x.describe()


# In[49]:


# SVM で学習

from sklearn.model_selection import train_test_split

train_size = 3000
test_size = 100

data_train, data_test, label_train, label_test = train_test_split(train_data_x, train_data_y, test_size=test_size, train_size=train_size, random_state=1)


# clf = svm.SVC()
# clf.fit(train_data_x, train_data_y)
# prediction = clf.predict(train_data_x)
# accuracy_score = metrics.accuracy_score(train_data_y, prediction)
# print(accuracy_score)


# In[50]:


# グリッドサーチ
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import metrics


# param_grid = {"C": [10 ** i for i in range(-5, 6)],
#                 "kernel": ["linear", "rbf", "sigmoid"],
#                 "decision_function_shape": ["ovo", "ovr"]
#                }

param_grid = {"C": [2, 5, 10, 20, 50],
                "kernel": ["linear", "rbf", "sigmoid"],
                "decision_function_shape": ["ovo", "ovr"]
               }

model_grid = GridSearchCV(estimator=svm.SVC(random_state=1),
                 param_grid = param_grid,   
                 scoring="accuracy",  # metrics
                 cv = 5,              # cross-validation
                 n_jobs = -1)          # number of core

model_grid.fit(data_train, label_train)

model_grid_best = model_grid.best_estimator_ # best estimator
print("Best Model Parameter: ", model_grid.best_params_)
# Best Model Parameter:  {'C': 10, 'decision_function_shape': 'ovo', 'kernel': 'rbf'}


# In[54]:


print('Train score: {}'.format(model_grid_best.score(data_train, label_train)))
print('Cross Varidation score: {}'.format(model_grid.best_score_))
# Train score: 1.0
# Cross Varidation score: 0.9383333333333332


# In[55]:


prediction = model_grid_best.predict(data_train)
co_mat = metrics.confusion_matrix(label_train, prediction)
print(co_mat)


# In[56]:


# 全データを使って学習する
clf = svm.SVC(C=5, decision_function_shape="ovo", kernel="rbf")
clf.fit(train_data_x, train_data_y)
prediction = clf.predict(train_data_x)
accuracy_score = metrics.accuracy_score(train_data_y, prediction)
print(accuracy_score)


# In[57]:


co_mat = metrics.confusion_matrix(train_data_y, prediction)
print(co_mat)


# In[58]:


prediction = clf.predict(test_data)
output = pd.DataFrame({"ImageId" : np.arange(1, 28000+1), "Label":prediction})
output.head()


# In[59]:


output.to_csv('digit_recognizer_SVM3.csv', index=False)
print("Your submission was successfully saved!")

