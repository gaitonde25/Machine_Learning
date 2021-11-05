#!/usr/bin/env python
# coding: utf-8

# In[53]:


# Do not make any changes in this cell
# Simply execute it and move on

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
ans = [0]*8


# In[54]:


# The exercise uses Boston housing dataset which is an inbuilt dataset of scikit learn.
# Run the cell below to import and get the information about the data.

# Do not make any changes in this cell.
# Simply execute it and move on

from sklearn.datasets import load_boston
boston=load_boston()
boston


# In[55]:


# Creating a dataframe

# Do not make any changes in this cell
# Simply execute it and move on

boston_df=pd.DataFrame(boston['data'], columns=boston['feature_names'])
boston_df['target'] = pd.DataFrame(boston['target'])
boston_df


# In[56]:


# Question 1: Find the mean of the "target" values in the dataframe (boston_df)
#             Assign the answer to ans[0]
#             eg. ans[0] = 24.976534890123 (if mean obtained = 24.976534890123)


# In[57]:


# Your Code: Enter your Code below
mean_T = boston_df['target'].mean()


# In[58]:


#1 mark
ans[0] = mean_T


# In[59]:


# Just to get a look into distribution of data into datasets
# Plot a histogram for boston_df

for col in boston_df.columns:
  boston_df.hist(col)


# **Splitting the data using train_test_split from sklearn library**

# In[60]:


# Import machine learning libraries  for train_test_split

from sklearn.model_selection import train_test_split

# Split the data into X and Y

X = pd.DataFrame(boston_df,columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
       'TAX', 'PTRATIO', 'B', 'LSTAT'])
Y = pd.DataFrame(boston_df,columns=['target'])

# Spliting our data further into train and test (train-90% and test-10%)
# Use (randon_state = 42) in train_test_split, so that your answer can be evaluated

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, train_size=0.9, random_state= 42)


# **LINEAR REGRESSION**

# In[61]:


# Question 2: Find mean squared error on the test set and the linear regression intercept(b)  
#             Assign the answer to ans[0] in the form of a list 
#             eg. ans[1] = [78.456398468,34.276498234098] 
#                  here , mean squared error             = 78.456398468
#                         linear regression intercept(b) = 34.276498234098


# In[62]:


# Fit a linear regression model on the above training data and find MSE over the test set.
# Your Code: Enter your Code below
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear_reg_model = LinearRegression().fit(X_train, Y_train)
mse = mean_squared_error(linear_reg_model.predict(X_test), Y_test)
b = linear_reg_model.intercept_


# In[63]:


# 2 marks
ans[1] = [mse, b.item()]


# **RIDGE REGRESSION**

# In[64]:


# Question 3: For what value of lambda (alpha)(in the list[0.5,1,5,10,50,100]) will we have least value of the mean squared error of testing set 
#             Take lambda (alpha) values as specified i.e. [0.5,1,5,10,50,100]
#             Assign the answer to ans[2]  
#             eg. ans[1] = 5  (if  lambda(alpha)=5)


# In[65]:


# Your Code: Enter your Code below
from sklearn.linear_model import Ridge

min_mse = float('inf')
target_alpha = None
ridge_models = []

for alpha in [0.5, 1, 5, 10, 50, 100]:
  ridge_model = Ridge(alpha=alpha)
  ridge_model.fit(X_train, Y_train)
  ridge_models.append(ridge_model)
  curr_mse = mean_squared_error(ridge_model.predict(X_test), Y_test)
  if curr_mse < min_mse:
    target_alpha = alpha
    min_mse = curr_mse


# In[66]:


#1 mark
ans[2] = target_alpha


# In[67]:


# Question 4: Find mean squared error on the test set and the Ridge regression intercept(b)
#             Use the lamba(alpha) value obtained from question-3 
#             Assign the answer to ans[3] in the form of a list 
#             eg. ans[3] = [45.456398468,143.276498234098] 
#                  here , mean squared error             = 45.456398468
#                         Ridge regression intercept(b) = 143.276498234098


# In[68]:


# Your Code: Enter your Code below
target_ridge_model = Ridge(alpha = target_alpha)
target_ridge_model.fit(X_train, Y_train)
mse_ = mean_squared_error(target_ridge_model.predict(X_test), Y_test)
b_ = target_ridge_model.intercept_


# In[69]:


# 2 marks
ans[3] = [mse_, b_.item()]


# In[70]:


# Plot the coefficient of the features( CRIM , INDUS , NOX ) with respective to  the lambda values specified [0.5,1,5,10,50,100]
# Enter your code below
CRIM_COEF = [model.coef_[0][0] for model in ridge_models]
INDUS_COEF = [model.coef_[0][2] for model in ridge_models]
NOX_COEF = [model.coef_[0][4] for model in ridge_models]

plt.plot([0.5, 1, 5, 10, 50, 100], CRIM_COEF, 'x', label='CRIM coefficients')
plt.title('CRIM coefficients')
plt.legend()
plt.show()

plt.plot([0.5, 1, 5, 10, 50, 100], INDUS_COEF, 'o', label='INDUS coefficients')
plt.title('INDUS coefficients')
plt.legend()
plt.show()

plt.plot([0.5, 1, 5, 10, 50, 100], NOX_COEF, '*', label='NOX coefficients')
plt.title('NOX coefficients')
plt.legend()
plt.show()


# **LASSO REGRESSION**

# In[71]:


# Question 5: For lambda (alpha)=1 find the lasso regression intercept and the test set mean squared error
#             Assign the answer to ans[4] in the form of a list
#             eg. ans[4] = [35.456398468,14.276498234098]
#                  here , mean squared error             = 35.456398468
#                         lasso regression intercept(b) = 14.276498234098


# In[72]:


# Your Code: Enter your Code below
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha = target_alpha)
lasso_model.fit(X_train, Y_train)
lasso_mse = mean_squared_error(lasso_model.predict(X_test), Y_test)
lasso_b = lasso_model.intercept_


# In[73]:


#2 mark
ans[4] = [lasso_mse, lasso_b.item()] 


# In[74]:


# Question 6: Find the most  important feature  in the data set i.e. which feature coefficient is further most non zero if lambda is increased gradually
#             let CRIM=1,	ZN=2, INDUS=3,	CHAS=4,	NOX=5,	RM=6,	AGE=7,	DIS=8,	RAD=9,	TAX=10,	PTRATIO=11,	B=12,	LSTAT=13
#              eg. if your answer is "CHAS"
#                   then your answer should be ans[5]=4


# In[75]:


# Your Code: Enter your Code below
names = boston_df.drop('target', axis = 1).columns

lasso_coef = lasso_model.fit(X_train, Y_train).coef_
plt.plot(range(len(names)),lasso_coef)
plt.xticks(range(len(names)),names,rotation = 60)
plt.ylabel('Coefficients')
plt.show()


# In[76]:


#2 marks
ans[5] = 6


# Run the below cell only once u complete answering all the above answers 
# 

# In[77]:


##do not change this code
import json
ans = [str(item) for item in ans]

filename = "group15_mohanlals1211973@gmail.com_Harsh_Sharma_RidgeRegression"

# Eg if your name is Saurav Joshi and email id is sauravjoshi123@gmail.com, filename becomes
# filename = sauravjoshi123@gmail.com_Saurav_Joshi_LinearRegression


# ## Do not change anything below!!
# - Make sure you have changed the above variable "filename" with the correct value. Do not change anything below!!

# In[82]:


from importlib import import_module
import os
from pprint import pprint

findScore = import_module('findScore')
response = findScore.main(ans)
response['details'] = filename
with open(f'evaluation_{filename}.json', 'w') as outfile:
    json.dump(response, outfile)
pprint(response)

