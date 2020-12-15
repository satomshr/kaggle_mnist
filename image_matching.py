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


plt.imshow(train_data.iloc[0, 1:].values.reshape(28, 28), cmap='gray')
plt.show()


# In[8]:


mnist_list = []

for i in range(10):
    mnist_list.append([])

for i in range(len(train_data)):
    mnist_list[train_data.iloc[i, 0]].append(train_data.iloc[i, 1:])
    

# h_num = 10
# w_num = 11

# fig = plt.figure(figsize=(h_num, w_num))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)

# for j in range(h_num):
#     for i in range(w_num):
#         ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[])
#         ax.imshow(mnist_list[j][i].values.reshape((28, 28)), cmap='gray')

# plt.show()


# In[9]:


# mnist_list[0][0]


# In[10]:


mnist_df = []
for i in range(10):
    mnist_df.append(pd.DataFrame(mnist_list[i]))


# In[11]:


h_num = 10
w_num = 11

fig = plt.figure(figsize=(h_num, w_num))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)

for j in range(h_num):
    for i in range(w_num):
        ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[])
        ax.imshow(mnist_df[j].iloc[i].values.reshape((28, 28)), cmap='gray')


plt.show()


# In[12]:


# average each image

mean_image = []
for i in range(10):
    mean_image.append(mnist_df[i].mean())
        
h_num = 10
w_num = 1

fig = plt.figure(figsize=(h_num*8, w_num*8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)

for j in range(h_num):
    ax = fig.add_subplot(h_num, w_num, j*w_num + 1, xticks=[], yticks=[])
    ax.imshow(mean_image[j].values.reshape((28, 28)), cmap='gray')


plt.show()


# In[13]:


mean_image[0].describe()


# In[14]:


train_data.iloc[0, 1:].describe()


# In[15]:


# https://pythondatascience.plavox.info/scikit-learn/%E5%9B%9E%E5%B8%B0%E3%83%A2%E3%83%87%E3%83%AB%E3%81%AE%E8%A9%95%E4%BE%A1

from sklearn.metrics import mean_squared_error

for i in range(10):
    mse = mean_squared_error(mean_image[i], train_data.iloc[0, 1:]) 
    print("%2d ; %f" % (i, mse))


# In[16]:


mse = []
for i in range(10):
    mse.append( mean_squared_error(mean_image[i], train_data.iloc[0, 1:]) )

print(np.argmin(mse))


# In[17]:


# predict
prediction = []

for j in range(len(train_data)):
    mse = []
    for i in range(10):
        mse.append(mean_squared_error(mean_image[i], train_data.iloc[j, 1:]))
        
    prediction.append(np.argmin(mse))

prediction


# In[18]:


# check prediction
from sklearn import metrics

accuracy_score = metrics.accuracy_score(train_data["label"], prediction)
print(accuracy_score)


# In[20]:


co_mat = metrics.confusion_matrix(train_data["label"], prediction)
print(co_mat)


# In[21]:


# prediction
prediction = []

for j in range(len(test_data)):
    mse = []
    for i in range(10):
        mse.append(mean_squared_error(mean_image[i], test_data.iloc[j]))
        
    prediction.append(np.argmin(mse))

output = pd.DataFrame({"ImageId" : np.arange(1, 28000+1), "Label":prediction})
output.head()


# In[22]:


output.to_csv('image_matching.csv', index=False)
print("Your submission was successfully saved!")
# score ; 0.80789

