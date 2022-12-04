#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import  DecisionTreeRegressor
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# In[24]:


df = pd.read_excel(r'DS - Assignment Part 1 data set.xlsx')
df.head()


# In[25]:


df.isnull().sum()


# In[26]:


df['Transaction date'].unique()


# In[27]:


df.corr()


# In[28]:


df.shape


# In[29]:


df.describe()


# In[30]:


X = df.drop(['TARGET(House price of unit area)'], axis = 1)
X


# In[31]:


y = df['TARGET(PRICE_IN_LACS)']
y


# In[32]:


#if your dependent variable contains categorical data, it is not compulsory to encode them
#for your independent variables, do not use label encoder to encode your
#1. one hot encoder is used to encode categorical data without inherent order
#2. ordinal encoder is used to encode categories with inherent order(where a category superceeds another category)

#we do not use label encoders to encode our independent features
#(especially when the categorical data contains more than two unique categories)
#becauseyour models will assume some inherent order


# In[33]:


X = pd.get_dummies(X, columns = ['POSTED_BY', 'BHK_OR_RK'], drop_first = True)
X


# In[34]:


scaler = MinMaxScaler()
X_scaler = scaler.fit_transform(X)
X_scaler


# In[35]:


df.shape


# In[36]:


X_train, X_test, y_train, y_test=train_test_split(X_scaler, y,  test_size = 0.3, random_state = 1)


# In[37]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[38]:


lr_model.score(X_train, y_train)


# In[39]:


knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)


# In[40]:


knn_model.score(X_test, y_test)


# In[41]:


knn_model.score(X_train, y_train)


# In[42]:


dt_model = DecisionTreeRegressor(max_depth = 5)
dt_model.fit(X_train, y_train)


# In[43]:


dt_model.score(X_test, y_test)


# In[44]:


dt_model.score(X_train, y_train)


# In[ ]:


testdf = pd.read_csv(r'/kaggle/input/house-price-prediction-challenge/test.csv')
testdf.head()


# In[ ]:


testdf.isnull().sum()


# In[ ]:


Xtest = testdf.drop(['ADDRESS'], axis = 1)
Xtest


# In[ ]:


Xtest_encoded = pd.get_dummies(Xtest, columns = ['POSTED_BY', 'BHK_OR_RK'], drop_first = True)
Xtest_encoded


# In[ ]:


Xtest_encoded_scaled = scaler.transform(Xtest_encoded)
Xtest_encoded_scaled


# In[ ]:


test_y_pred = dt_model.predict(Xtest_encoded_scaled)


# In[ ]:


test_y_result_df = pd.DataFrame(test_y_pred, columns = ['TARGET(PRICE_IN_LACS)'])
test_y_result_df


# In[46]:


test_y_result_df.to_csv(r'result.csv', index = None)


# In[ ]:





# In[ ]:




