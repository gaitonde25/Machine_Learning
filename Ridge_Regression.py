import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# The exercise uses Boston housing dataset which is an inbuilt dataset of scikit learn.

from sklearn.datasets import load_boston
boston=load_boston()
boston

# Creating a dataframe

boston_df=pd.DataFrame(boston['data'], columns=boston['feature_names'])
boston_df['target'] = pd.DataFrame(boston['target'])
boston_df

# Find the mean of the "target" values in the dataframe (boston_df)
mean_T = boston_df['target'].mean()

# Just to get a look into distribution of data into datasets
# Plot a histogram for boston_df

for col in boston_df.columns:
  boston_df.hist(col)


# **Splitting the data using train_test_split from sklearn library**

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

# Find mean squared error on the test set and the linear regression intercept(b)  

# Fit a linear regression model on the above training data and find MSE over the test set.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

linear_reg_model = LinearRegression().fit(X_train, Y_train)
mse = mean_squared_error(linear_reg_model.predict(X_test), Y_test)
b = linear_reg_model.intercept_

# **RIDGE REGRESSION**
# For what value of lambda (alpha)(in the list[0.5,1,5,10,50,100]) will we have least value of the mean squared error of testing set 
# Take lambda (alpha) values as specified i.e. [0.5,1,5,10,50,100]


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

# Find mean squared error on the test set and the Ridge regression intercept(b)

target_ridge_model = Ridge(alpha = target_alpha)
target_ridge_model.fit(X_train, Y_train)
mse_ = mean_squared_error(target_ridge_model.predict(X_test), Y_test)
b_ = target_ridge_model.intercept_


# Plot the coefficient of the features( CRIM , INDUS , NOX ) with respective to  the lambda values specified [0.5,1,5,10,50,100]
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

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha = target_alpha)
lasso_model.fit(X_train, Y_train)
lasso_mse = mean_squared_error(lasso_model.predict(X_test), Y_test)
lasso_b = lasso_model.intercept_


# Question 6: Find the most  important feature  in the data set i.e. which feature coefficient is further most non zero if lambda is increased gradually

names = boston_df.drop('target', axis = 1).columns

lasso_coef = lasso_model.fit(X_train, Y_train).coef_
plt.plot(range(len(names)),lasso_coef)
plt.xticks(range(len(names)),names,rotation = 60)
plt.ylabel('Coefficients')
plt.show()




