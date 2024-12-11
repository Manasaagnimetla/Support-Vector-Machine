#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[17]:


data= pd.read_csv(r"C:\Users\Lenovo\OneDrive - RandomTrees\Desktop\Iris.csv")
data


# In[18]:


from matplotlib import pyplot as plt


# In[21]:


plt.scatter(data['SepalLengthCm'],data['PetalLengthCm'],color='green',marker='+')


# In[22]:


plt.scatter(data['SepalWidthCm'],data['PetalWidthCm'],color='green',marker='+')


# In[23]:


from sklearn.model_selection import train_test_split


# In[26]:


x= data.drop(['Species'],axis='columns')


# In[27]:


x


# In[29]:


y=data['Species']


# In[30]:


y


# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[32]:


x_train


# In[33]:


y_train


# In[35]:


from sklearn.svm import SVC


# In[37]:


model=SVC()


# In[38]:


model.fit(x_train,y_train)


# In[39]:


model.score(x_test,y_test)


# In[ ]:




