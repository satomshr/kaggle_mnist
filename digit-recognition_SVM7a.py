#!/usr/bin/env python
# coding: utf-8

# # Use SVM.svc, creating new features

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## load data

# In[ ]:


train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# In[ ]:


train_data_len = len(train_data)
test_data_len = len(test_data)
print("Length of train_data ; {}".format(train_data_len))
print("Length of test_data ; {}".format(test_data_len))


# - Length of train_data ; 42000
# - Length of test_data ; 28000

# In[ ]:


train_data_y = train_data["label"]
train_data_x = train_data.drop(columns="label")
train_data_x.head()


# ## create new features
# ### concatenate and transpose

# In[ ]:


df = pd.concat([train_data_x, test_data])
df_T = df.T
df_T.describe()


# In[ ]:


df_T.head()


# In[ ]:


df_T["no"] = range(len(df_T))
df_T.head()


# In[ ]:


df_T.tail()


# ### mean and std of all area

# In[ ]:


df_T.loc["a_mean"] = df_T.mean()
df_T.loc["a_std"] = df_T.std()
df_T.tail()


# ### mean and std of 1/2 area

# In[ ]:


# horizontal

for i in range(2):
    q = 'no < 28*28/2' if i == 0 else 'no >= 28*28/2'
    df_T.loc["b{}_mean".format(i)] = df_T[:784].query(q).mean()
    df_T.loc["b{}_std".format(i)] = df_T[:784].query(q).std()

    
df_T.tail()


# In[ ]:


# vertical

for i in range(2):
    q = 'no % 28 < 14' if i == 0 else 'no % 28 >= 14'
    df_T.loc["c{}_mean".format(i)] = df_T[:784].query(q).mean()
    df_T.loc["c{}_std".format(i)] = df_T[:784].query(q).std()

    
df_T.tail()


# ### mean and std of 1/4 area

# In[ ]:


for i in range(2):
    qi = 'no < 28*28/2' if i == 0 else 'no >= 28*28/2'
    for j in range(2):
        qj = 'no % 28 < 14' if j == 0 else 'no % 28 >= 14'
        q = qi + " & " + qj
        num = i * 2 + j
        df_T.loc["d{}_mean".format(num)] = df_T[:784].query(q).mean()
        df_T.loc["d{}_std".format(num)] = df_T[:784].query(q).std()

        
df_T.tail()


# ### mean and std of 1/9 area

# In[ ]:


for i in range(3):
    if i == 0:
        qi = 'no < 262'
    elif i == 1:
        qi = "262 <= no < 522"
    else:
        qi = "522 <= no < 784"
        
    for j in range(3):
        if j == 0:
            qj = 'no % 28 < 9'
        elif j == 1:
            qj = '9 <= no % 28 < 18'
        else:
            qj = '18 <= no % 28'
        
        q = qi + " & " + qj
        num = i * 3 + j
        df_T.loc["e{}_mean".format(num)] = df_T[:784].query(q).mean()
        df_T.loc["e{}_std".format(num)] = df_T[:784].query(q).std()

        
df_T.tail(20)


# ### mean and std of 1/16 area

# In[ ]:


for i in range(4):
    qi = '{0} <= no < {1}'.format(28*28/4*i, 28*28/4*(i+1))
        
    for j in range(4):
        qj = '{0} <= no % 28 < {1}'.format(28/4*j, 28/4*(j+1))
        
        q = qi + " & " + qj
        num = i * 4 + j
        df_T.loc["f{}_mean".format(num)] = df_T[:784].query(q).mean()
        df_T.loc["f{}_std".format(num)] = df_T[:784].query(q).std()

        
df_T.tail(32)


# ### drop "no" and re-transpose

# In[ ]:


df_T.drop(columns="no", inplace=True)
df = df_T.T
df.head()


# ## drop columns if all values are the same

# In[ ]:


drop_col = []

for c in df.columns:
    col_max = df[c].max()
    col_min = df[c].min()
    if col_max == col_min:
        drop_col.append(c)
        
print("# of dropping columns ; {}".format(len(drop_col)))
        
df.drop(drop_col, axis=1, inplace=True)

df.head()


# - number of dropping columns ; 65

# ## scaling

# In[ ]:


from sklearn import preprocessing

mmscaler = preprocessing.MinMaxScaler()

# data = pd.concat([train_data_x, test_data])

mmscaler.fit(df)

# mmscaler.transform() is np.ndarray, so change it to pd.DataFrame
df_scaled = pd.DataFrame(mmscaler.transform(df), columns=df.columns, index=df.index)

df_scaled.head()


# In[ ]:


df_scaled.describe()


# ## separate df_scaled into train_data and test data

# In[ ]:


train_data_x = df_scaled[:train_data_len]
test_data = df_scaled[train_data_len:]


# In[ ]:


train_data_x.describe()


# In[ ]:


train_data_x.head()


# In[ ]:


test_data.describe()


# In[ ]:


test_data.head()


# ## svm.SVC ; parameter optimization
# ### (1) try wide range of C and gamma roughly, train_data : 3000

# In[ ]:


from sklearn.model_selection import train_test_split
train_data_x_sub, x_test, train_data_y_sub, y_test = train_test_split(train_data_x, train_data_y,
                                                                      train_size=8000, test_size=100,
                                                                      random_state=1)
train_data_x_sub.info()


# In[ ]:


# SVM1

# from sklearn.model_selection import GridSearchCV
# from sklearn import svm
# from sklearn import metrics

# train_size : 3000

# param_grid = {"C": [10 ** i for i in range(-5, 6, 2)],
#               "gamma" : [10 ** i for i in range(-5, 6, 2)]
#              }

# model_grid = GridSearchCV(estimator=svm.SVC(kernel="rbf", decision_function_shape="ovo", random_state=1),
#                  param_grid = param_grid,   
#                  scoring = "accuracy",  # metrics
#                  verbose = 2,
#                  cv = 4)              # cross-validation

# model_grid.fit(train_data_x_sub, train_data_y_sub)
# model_grid_best = model_grid.best_estimator_ # best estimator
# print("Best Model Parameter: ", model_grid.best_params_)


# - Best model parameter : {'C': 10, 'gamma': 0.001}
# - 1 calculation : 13 - 14s

# In[ ]:


# print('Train score: {}'.format(model_grid_best.score(train_data_x_sub, train_data_y_sub)))
# print('Cross Varidation score: {}'.format(model_grid.best_score_))


# - Train score: 0.9593333333333334
# - Cross Varidation score: 0.9063333333333333

# In[ ]:


# means = model_grid.cv_results_['mean_test_score']
# stds = model_grid.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, model_grid.cv_results_['params']):
#     print("%0.4f (+/-%0.04f) for %r" % (mean, std * 2, params))


# In[ ]:


# prediction = model_grid_best.predict(train_data_x_sub)
# co_mat = metrics.confusion_matrix(train_data_y_sub, prediction)
# print(co_mat)


# In[ ]:


# print('Total Train score: {}'.format(model_grid_best.score(train_data_x, train_data_y)))
# print('Cross Varidation score: {}'.format(model_grid.best_score_))


# - Total Train score: 0.9251904761904762

# In[ ]:


# prediction = model_grid_best.predict(train_data_x)
# co_mat = metrics.confusion_matrix(train_data_y, prediction)
# print(co_mat)


# ### (2) try C : 0.3 - 300, and gamma : 0.0001 : 0.03, train_data : 3000

# In[ ]:


# SVM2

# from sklearn.model_selection import GridSearchCV
# from sklearn import svm
# from sklearn import metrics

# train_data : 3000

# param_grid = {"C": [0.3, 1, 3, 10, 30, 100, 300],
#               "gamma" : [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]
#             }

# model_grid = GridSearchCV(estimator=svm.SVC(kernel="rbf", decision_function_shape="ovo", random_state=1),
#                  param_grid = param_grid,   
#                  scoring = "accuracy",  # metrics
#                  verbose = 2,
#                  cv = 5)              # cross-validation

# model_grid.fit(train_data_x_sub, train_data_y_sub)
# model_grid_best = model_grid.best_estimator_ # best estimator
# print("Best Model Parameter: ", model_grid.best_params_)


# - Best model parameter : {'C': 3, 'gamma': 0.03}
# - 1 calculation : 9 - 16s

# In[ ]:


# print('Train score: {}'.format(model_grid_best.score(train_data_x_sub, train_data_y_sub)))
# print('Cross Varidation score: {}'.format(model_grid.best_score_))


# - Train score: 1.0
# - Cross Varidation score: 0.9443333333333334

# In[ ]:


# means = model_grid.cv_results_['mean_test_score']
# stds = model_grid.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, model_grid.cv_results_['params']):
#     print("%0.4f (+/-%0.04f) for %r" % (mean, std * 2, params))


# In[ ]:


# from sklearn import metrics

# prediction = model_grid_best.predict(train_data_x_sub)
# co_mat = metrics.confusion_matrix(train_data_y_sub, prediction)
# print(co_mat)


# In[ ]:


# print('Total Train score: {}'.format(model_grid_best.score(train_data_x, train_data_y)))
# print('Cross Varidation score: {}'.format(model_grid.best_score_))


# - Total Train score: 0.9587857142857142

# In[ ]:





# ### (3) missing
# ### (4) try C : 3 - 300, and gamma : 0.025 : 0.05, train_data : 8000

# In[ ]:


# SVM4

# from sklearn.model_selection import GridSearchCV
# from sklearn import svm
# from sklearn import metrics

# train_data : 8000

# param_grid = {"C": [3, 10, 30, 100, 300],
#               "gamma" : [0.025, 0.03, 0.04, 0.05]
#             }

# model_grid = GridSearchCV(estimator=svm.SVC(kernel="rbf", decision_function_shape="ovo", random_state=1),
#                  param_grid = param_grid,   
#                  scoring = "accuracy",  # metrics
#                  verbose = 2,
#                  cv = 5)              # cross-validation

# model_grid.fit(train_data_x_sub, train_data_y_sub)
# model_grid_best = model_grid.best_estimator_ # best estimator
# print("Best Model Parameter: ", model_grid.best_params_)


# - Best model parameter : {'C': 10, 'gamma': 0.025}
# - 1 calculation time : 30s - 1.5min

# In[ ]:


# print('Train score: {}'.format(model_grid_best.score(train_data_x_sub, train_data_y_sub)))
# print('Cross Varidation score: {}'.format(model_grid.best_score_))


# - Train score: 1.0
# - Cross Varidation score: 0.9695

# In[ ]:


# means = model_grid.cv_results_['mean_test_score']
# stds = model_grid.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, model_grid.cv_results_['params']):
#     print("%0.4f (+/-%0.04f) for %r" % (mean, std * 2, params))


# In[ ]:


# from sklearn import metrics

# prediction = model_grid_best.predict(train_data_x_sub)
# co_mat = metrics.confusion_matrix(train_data_y_sub, prediction)
# print(co_mat)


# In[ ]:


# print('Total Train score: {}'.format(model_grid_best.score(train_data_x, train_data_y)))
# print('Cross Varidation score: {}'.format(model_grid.best_score_))


# - Total Train score: 0.9737857142857143

# In[ ]:





# (5) try C : 5 - 20, and gamma : 0.015 : 0.028, train_data : 8000

# In[ ]:


# SVM5

# from sklearn.model_selection import GridSearchCV
# from sklearn import svm
# from sklearn import metrics

# train_data : 8000

# param_grid = {"C": [5, 8, 10, 15, 20],
#               "gamma" : [0.015, 0.02, 0.024, 0.028]
#             }

# model_grid = GridSearchCV(estimator=svm.SVC(kernel="rbf", decision_function_shape="ovo", random_state=1),
#                  param_grid = param_grid,   
#                  scoring = "accuracy",  # metrics
#                  verbose = 2,
#                  cv = 5)              # cross-validation

# model_grid.fit(train_data_x_sub, train_data_y_sub)
# model_grid_best = model_grid.best_estimator_ # best estimator
# print("Best Model Parameter: ", model_grid.best_params_)


# - Best model parameter : {'C': 5, 'gamma': 0.028}
# - 1 calculation time : around 30s

# In[ ]:


# print('Train score: {}'.format(model_grid_best.score(train_data_x_sub, train_data_y_sub)))
# print('Cross Varidation score: {}'.format(model_grid.best_score_))


# - Train score: 1.0
# - Cross Varidation score: 0.969375

# In[ ]:


# means = model_grid.cv_results_['mean_test_score']
# stds = model_grid.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, model_grid.cv_results_['params']):
#     print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))


# In[ ]:


# from sklearn import metrics

# prediction = model_grid_best.predict(train_data_x_sub)
# co_mat = metrics.confusion_matrix(train_data_y_sub, prediction)
# print(co_mat)


# In[ ]:


# print('Total Train score: {}'.format(model_grid_best.score(train_data_x, train_data_y)))
# print('Cross Varidation score: {}'.format(model_grid.best_score_))


# - Total Train score: 0.9738095238095238

# In[ ]:





# ## learning with optimal parameters using all train_data

# In[ ]:


# learning with optimal parameters using all data

from sklearn import svm
from sklearn import metrics

clf = svm.SVC(C=5, gamma=0.028, decision_function_shape="ovo", kernel="rbf", verbose=2)
clf.fit(train_data_x, train_data_y)
prediction = clf.predict(train_data_x)
accuracy_score = metrics.accuracy_score(train_data_y, prediction)
print(accuracy_score)


# - Accuracy : 0.9999761904761905

# In[ ]:


co_mat = metrics.confusion_matrix(train_data_y, prediction)
print(co_mat)


# In[ ]:


prediction = clf.predict(test_data)
output = pd.DataFrame({"ImageId" : np.arange(1, 28000+1), "Label":prediction})
output.head()


# In[ ]:


output.to_csv('digit_recognizer_SVM7a.csv', index=False)
print("Your submission was successfully saved!")

