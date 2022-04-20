#!/usr/bin/env python
# coding: utf-8

# # Feature scaling with sklearn - Exercise

# You are given a real estate dataset. 
# 
# Real estate is one of those examples that every regression course goes through as it is extremely easy to understand and there is a (almost always) certain causal relationship to be found.
# 
# The data is located in the file: 'real_estate_price_size_year.csv'. 
# 
# You are expected to create a multiple linear regression (similar to the one in the lecture), using the new data. This exercise is very similar to a previous one. This time, however, **please standardize the data**.
# 
# Apart from that, please:
# -  Display the intercept and coefficient(s)
# -  Find the R-squared and Adjusted R-squared
# -  Compare the R-squared and the Adjusted R-squared
# -  Compare the R-squared of this regression and the simple linear regression where only 'size' was used
# -  Using the model make a prediction about an apartment with size 750 sq.ft. from 2009
# -  Find the univariate (or multivariate if you wish - see the article) p-values of the two variables. What can you say about them?
# -  Create a summary table with your findings
# 
# In this exercise, the dependent variable is 'price', while the independent variables are 'size' and 'year'.
# 
# Good luck!

# ## Import the relevant libraries

# In[11]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# ## Load the data

# In[12]:


data = pd.read_csv('real_estate_price_size_year.csv')


# In[13]:


data.head()


# In[14]:


data.describe()


# ## Create the regression

# ### Declare the dependent and the independent variables

# In[15]:


x = data[['size','year']]
y = data['price']


# ### Scale the inputs

# In[19]:


scaler = StandardScaler()
scaler.fit(x)


# In[22]:


x_scaled = scaler.transform(x)


# In[23]:


x_scaled


# ### Regression

# In[24]:


reg = LinearRegression()
reg.fit(x_scaled,y)


# ### Find the intercept

# In[25]:


reg.intercept_


# ### Find the coefficients

# In[26]:


reg.coef_


# ### Calculate the R-squared

# In[32]:


reg.score(x_scaled,y)


# ### Calculate the Adjusted R-squared

# In[33]:


x.shape


# In[35]:


r2 = reg.score(x_scaled,y)
n  = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1-(1 - r2) * (n-1)/(n-p-1)
adjusted_r2


# ### Compare the R-squared and the Adjusted R-squared

# Answer...
# r2 = 0.7764803683276793
# adjusted_r2= 0.77187171612825
# 
# The difference is not really significant, that's mean that we are being penalized for the extra variable

# ### Compare the Adjusted R-squared with the R-squared of the simple linear regression

# Answer...
# Multiple Linear Regression adjusted_r2= 0.77187171612825
# Simple Linear Regression adjusted_r2 = 0.745
# 
# It means the year doesn't apport so much to the analysis

# ### Making predictions
# 
# Find the predicted price of an apartment that has a size of 750 sq.ft. from 2009.

# In[54]:


new_data = [[750,2009]]
new_data_scaled = scaler.transform(new_data)


# In[56]:


reg.predict(new_data_scaled)


# ### Calculate the univariate p-values of the variables

# In[57]:


from sklearn.feature_selection import f_regression


# In[58]:


f_regression(x,y)


# In[59]:


p_values = f_regression(x,y)[1]
p_values


# In[60]:


p_values.round(3)


# ### Create a summary table with your findings

# In[63]:


reg_summary = pd.DataFrame(data=x.columns.values, columns=['features'])
reg_summary


# In[64]:


reg_summary['Coefficients'] = reg.coef_
reg_summary['p_values'] = p_values.round(3)
reg_summary


# Answer...
# we have been penalized for the variable year
# Note that this dataset is extremely clean and probably artificially created, therefore standardization does not really bring any value to it.
