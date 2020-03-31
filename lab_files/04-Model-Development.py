# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:11:02 2020

@author: ahmed

Model Development

Data Analytics, we often use Model Development to help us predict future observations 
from the data we have.

A Model will help us understand the exact relationship between different 
variables and how these variables are used to predict the result.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load data and store in dataframe df:
# path of data 
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

""" ########$ 1. Linear Regression and Multiple Linear Regression
######Simple Linear Regression.
Simple Linear Regression is a method to help us understand the relationship between two variables:

# The predictor/independent variable (X)
# The response/dependent variable (that we want to predict)(Y)
The result of Linear Regression is a linear function that predicts the response (dependent) 
variable as a function of the predictor (independent) variable.

                    𝑌:𝑅𝑒𝑠𝑝𝑜𝑛𝑠𝑒 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒
                    𝑋:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒𝑠
Linear function:
                            𝑌ℎ𝑎𝑡=𝑎+𝑏𝑋
 
        a refers to the intercept of the regression line0, in other words: the value of Y when X is 0
        b refers to the slope of the regression line, in other words: the value with which Y changes 
        when X increases by 1 unit
Lets load the modules for linear regression"""

from sklearn.linear_model import LinearRegression

# Create the linear regression object
lm = LinearRegression()
lm

"""How could Highway-mpg help us predict car price?
For this example, we want to look at how highway-mpg can help us predict car price. 
Using simple linear regression, we will create a linear function with "highway-mpg" 
as the predictor variable and the "price" as the response variable."""

X = df[['highway-mpg']]
Y = df['price']

# Fit the linear model using highway-mpg.
lm.fit(X,Y)
#  We can output a prediction 
Yhat=lm.predict(X)
Yhat[0:5] 

# What is the value of the intercept (a)?
lm.intercept_
# What is the value of the Slope (b)?
lm.coef_

"""What is the final estimated linear model we get?¶
As we saw above, we should get a final linear model with the structure:

𝑌ℎ𝑎𝑡=𝑎+𝑏𝑋
 
Plugging in the actual values we get:

price = 38423.31 - 821.73 x highway-mpg"""



"""Q2 Train the model using 'engine-size' as the independent 
variable and 'price' as the dependent variable?"""

lm1 = LinearRegression()
lm1 
lm1.fit(df[['engine-size']], df[['price']])
lm1

# Find the slope and intercept of the model?
# Slope 
lm1.coef_
# Intercept
lm1.intercept_

# What is the equation of the predicted line. You can use x and yhat or 'engine-size' or 'price'?

# using X and Y  
# Yhat=38423.31-821.733*X
# Price=38423.31-821.733*'engine-size'


"""############################## Multiple Linear Regression
What if we want to predict car price using more than one variable?

If we want to use more variables in our model to predict car price, 
we can use Multiple Linear Regression. Multiple Linear Regression is very similar to 
Simple Linear Regression, but this method is used to explain the relationship between 
one continuous response (dependent) variable and two or more predictor (independent) variables. 
Most of the real-world regression models involve multiple predictors. 
We will illustrate the structure by using four predictor variables, 
but these results can generalize to any integer:

                                            𝑌:𝑅𝑒𝑠𝑝𝑜𝑛𝑠𝑒 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒
                                    𝑋1:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 1
                                    𝑋2:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 2
                                    𝑋3:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 3
                                    𝑋4:𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑜𝑟 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 4
                                            𝑎:𝑖𝑛𝑡𝑒𝑟𝑐𝑒𝑝𝑡
                                    𝑏1:𝑐𝑜𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑡𝑠 𝑜𝑓 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 1
                                    𝑏2:𝑐𝑜𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑡𝑠 𝑜𝑓 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 2
                                    𝑏3:𝑐𝑜𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑡𝑠 𝑜𝑓 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 3
                                    𝑏4:𝑐𝑜𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑡𝑠 𝑜𝑓 𝑉𝑎𝑟𝑖𝑎𝑏𝑙𝑒 4
The equation is given by

𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋3+𝑏4𝑋4
 
From the previous section we know that other good predictors of price could be:

            Horsepower
            Curb-weight
            Engine-size
            Highway-mpg
Let's develop a model using these variables as the predictor variables."""

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
# Fit the linear model using the four above-mentioned variables.
lm.fit(Z, df['price'])
# What is the value of the intercept(a)?
lm.intercept_
# What are the values of the coefficients (b1, b2, b3, b4)?
lm.coef_
"""What is the final estimated linear model that we get?

As we saw above, we should get a final linear function with the structure:

                𝑌ℎ𝑎𝑡=𝑎+𝑏1𝑋1+𝑏2𝑋2+𝑏3𝑋3+𝑏4𝑋4
 
What is the linear function we get in this example?

Price = -15678.742628061467 + 52.65851272 x horsepower + 4.69878948 x curb-weight + 81.95906216
         x engine-size + 33.58258185 x highway-mpg"""


"""################ Create and train a Multiple Linear Regression model "lm2" 
    where the response variable is price, and the predictor variable is 'normalized-losses'
    and 'highway-mpg'."""
lm2 = LinearRegression()
lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])
# >Find the coefficient of the model?
lm2.coef_

"""###################### 2)  Model Evaluation using Visualization

Now that we've developed some models, how do we evaluate our models and 
how do we choose the best one? One way to do this is by using visualization.

import the visualization package: seaborn"""
# import the visualization package: seaborn
import seaborn as sns
#%matplotlib inline 

"""     Regression Plot 
When it comes to simple linear regression, an excellent way to visualize 
the fit of our model is by using regression plots.

This plot will show a combination of a scattered data points (a scatter plot), 
as well as the fitted linear regression line going through the data. 
This will give us a reasonable estimate of the relationship between the two variables, 
the strength of the correlation, as well as the direction (positive or negative correlation).

Let's visualize Horsepower as potential predictor variable of price:"""
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim(0,)

"""We can see from this plot that price is negatively correlated to highway-mpg, since the regression slope is negative. One thing to keep in mind when looking at a regression plot is to pay attention to how scattered the data points are around the regression line. This will give you a good indication of the variance of the data, and whether a linear model would be the best fit or not. If the data is too far off from the line, this linear model might not be the best model for this data. Let's compare this plot to the regression plot of "peak-rpm"."""

plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)

"""Comparing the regression plot of "peak-rpm" and "highway-mpg" we see that the points for "highway-mpg" are much closer to the generated line and on the average decrease. The points for "peak-rpm" have more spread around the predicted line, and it is much harder to determine if the points are decreasing or increasing as the "highway-mpg" increases."""


"""Given the regression plots above is "peak-rpm" or "highway-mpg" 
more strongly correlated with "price". Use the method ".corr()" to verify your answer."""

"""Answer
The variable "peak-rpm" has a stronger correlation with "price", it is approximate -0.704692
  compared to   "highway-mpg" which is approximate     -0.101616. 
    You can verify it using the following command:"""
    
    
df[["peak-rpm","highway-mpg","price"]].corr()




"""##### Residual Plot
A good way to visualize the variance of the data is to use a residual plot.

What is a residual?

The difference between the observed value (y) and the predicted value (Yhat) is called the residual (e). When we look at a regression plot, the residual is the distance from the data point to the fitted regression line.

So what is a residual plot?

    A residual plot is a graph that shows the residuals on the vertical y-axis 
    and the independent variable on the horizontal x-axis.

What do we pay attention to when looking at a residual plot?

We look at the spread of the residuals:

- If the points in a residual plot are randomly spread out around the x-axis, then a linear model is appropriate for the data. Why is that? Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data."""

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['highway-mpg'], df['price'])
plt.show()
"""What is this plot telling us?

We can see from this residual plot that the residuals are not randomly spread around the x-axis, which leads us to believe that maybe a non-linear model is more appropriate for this data."""

"""############# Multiple Linear Regression
How do we visualize a model for Multiple Linear Regression? This gets a bit more complicated because you can't visualize it with regression or residual plot.

One way to look at the fit of the model is by looking at the distribution plot: We can look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.

First lets make a prediction"""

Y_hat = lm.predict(Z)

plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()














##-------------------------------------------------------------------------------------##

"""Let X be a dataframe with 100 rows and 5 columns, let y be the target with 100 samples,
assuming all the relevant libraries and data have been imported, the following line of code has been executed:

LR = LinearRegression()

LR.fit(X, y)

yhat = LR.predict(X)

How many samples does yhat contain : ANSwer100"""

""" Although the predictor variables of Polynomial linear regression are not linear the relationship between the parameters or coefficients is linear. 
"""

"""Assume all the libraries are imported, y is the target and X is the features or dependent variables, consider the following lines of code:

Input = [('scale', StandardScaler()), ('model', LinearRegression())]

pipe = Pipeline(Input)

pipe.fit(X,y)

ypipe = pipe.predict(X)

What have we just done in the above code?

Answer:
    Standardize the data, then perform prediction using a linear regression model - correct"""

"""The larger the mean square error, the better your model has performed
    Answer: false"""
