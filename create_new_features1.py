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


train_data.iloc[0, 1:]


# In[8]:


plt.imshow(train_data.iloc[0, 1:].values.reshape(28, 28), cmap='Greys')
plt.show()


# In[9]:


# https://uchidama.hatenablog.com/entry/2017/12/19/183000

# h_num = 10
# w_num = 10

# fig = plt.figure(figsize=(h_num, w_num))
# fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)
# for i in range(h_num * w_num):
#     ax = fig.add_subplot(h_num, w_num, i + 1, xticks=[], yticks=[])
#     ax.imshow(train_data.iloc[i, 1:].values.reshape((28, 28)), cmap='gray')

# plt.show()


# In[10]:


mnist_list = []
for i in range(10):
    mnist_list.append([])

# for i in range(len(train_data)):
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


# In[11]:


# mnist_list[0][0]


# In[12]:


mnist_df = []
for i in range(10):
    mnist_df.append(pd.DataFrame(mnist_list[i]))


# In[13]:


h_num = 10
w_num = 11

fig = plt.figure(figsize=(h_num, w_num))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)

for j in range(h_num):
    for i in range(w_num):
        ax = fig.add_subplot(h_num, w_num, i + j*w_num + 1, xticks=[], yticks=[])
        ax.imshow(mnist_df[j].iloc[i].values.reshape((28, 28)), cmap='gray')


plt.show()


# In[14]:


# average each image

h_num = 10
w_num = 1

fig = plt.figure(figsize=(h_num*8, w_num*8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1.0, hspace=0.05, wspace=0.05)

for j in range(h_num):
    ax = fig.add_subplot(h_num, w_num, j*w_num + 1, xticks=[], yticks=[])
    ax.imshow(mnist_df[j].mean().values.reshape((28, 28)), cmap='gray')


plt.show()


# In[ ]:




