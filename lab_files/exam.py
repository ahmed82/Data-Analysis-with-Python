# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:38:56 2020

@author: ahmed
Final exam
"""


df.dropna(subset=["price"], axis=0)
#  Drop the “not a number” from the column price  - correct


# Q2 How would you provide many of the summery statistics for all the columns in the dataframe "df":
df.describe(include = "all")

# How would you find the shape of the dataframe df
df.shape

# What task does the following command to df.to_csv("A.csv") perform
Save the dataframe df to a csv file called "A.csv"

#5 What task does the following line of code perform:
df['peak-rpm'].replace(np.nan, 5,inplace=True)
# replace the not a number values with 5 in the column 'peak-rpm'

# 6 How do you "one hot encode" the column 'fuel-type' in the dataframe df
pd.get_dummies(df["fuel-type"]) 

# 8 What does the vertical axis in a scatter plot represent
# dependent variable

# 9 What does the horizontal axis in a scatter plot represent
# independent variable

# 10 If we have 10 columns and 100 samples how large is the output of df.corr()
#  10 x 10 - correct
    
# 11 what is the largest possible element resulting in the following operation "df.corr()"
# 1

# 12 if the Pearson Correlation of two variables is zero:
 the two variable have zero mean  
 the two variables are not correlated

# 13 if the p value of the Pearson Correlation is 1:

 the variables are correlated 
 the variables are correlated - 
  none of the above ------ corect


# 14 What does the following line of code do: lm = LinearRegression()
# create a linear regression object create a linear regression object - correct


# 15 If the predicted function is:
Yhat = a + b1 X1 + b2 X2 + b3 X3 + b4 X4
# The method is Multiple Linear Regression

# 16 What steps do the following lines of code perform:
Input=[('scale',StandardScaler()),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)

#  Standardize the data, then perform a prediction using a linear regression model using the features Z and targets y Standardize the data, then perform a prediction using a linear regression model using the features Z and targets y - correct

# 17 What is the maximum value of R^2 that can be obtained
# 1

# 18 We create a polynomial feature as follows "PolynomialFeatures(degree=2)", what is the order of the polynomial
# 2

# 19 You have a linear model the average R^2 value on your training data is 0.5, 
# you perform a 100th order polynomial transform on your data then use these values
# to train another model, your average R^2 is 0.99 which comment is correct
# Answer : the results on your training data is not the best indicator of how your model performs, you should use your test data to get a beter idea

# 20 You train a ridge regression model, you get a R^2 of 1 on your training
#        data and you get a R^2 of 0 on your validation data, what should you do:
# Answer:  Nothing your model performs flawlessly on your test data 























