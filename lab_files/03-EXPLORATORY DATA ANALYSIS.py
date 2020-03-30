# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:59:28 2020

@author: 1426391

EXPLORATORY DATA ANALYSIS
"""

# 1. Import Data from Module 2
import pandas as pd
import numpy as np

# Load data and store in dataset
path='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

"""
2. Analyzing Individual Feature Patterns using Visualization
To install seaborn we use the pip which is the python package manager.
"""
## %%capture
#! pip install seaborn

#Import visualization packages "Matplotlib" and "Seaborn", don't forget about "%matplotlib inline" 
#to plot in a Jupyter notebook.
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline 
#When visualizing individual variables, it is important to first understand what type of variable you are dealing with. 
#This will help us find the right visualization method for that variable.

# list the data types for each column
print(df.dtypes)

# for example, we can calculate the correlation between variables  of type "int64" or "float64" 
# sing the method "corr": df.corr()
# Find the correlation between the following columns: bore, stroke,compression-ratio , and horsepower.
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()  


"""
####### Continuous numerical variables:
Continuous numerical variables are variables that may contain any value within some range. Continuous numerical variables can have the type "int64" or "float64". A great way to visualize these variables is by using scatterplots with fitted lines.

In order to start understanding the (linear) relationship between an individual variable and the price. We can do this by using "regplot", which plots the scatterplot plus the fitted regression line for the data.

Let's see several examples of different linear relationships:
    ########################### Positive linear relationship
Let's find the scatterplot of "engine-size" and "price"
"""
# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)

"""
As the engine-size goes up, the price goes up: this indicates a positive direct correlation
between these two variables. Engine size seems like a pretty good predictor of price since 
the regression line is almost a perfect diagonal line.

We can examine the correlation between 'engine-size' and 'price' and see it's approximately 0.87
"""
df[["engine-size", "price"]].corr()

sns.regplot(x="highway-mpg", y="price", data=df)
"""
As the highway-mpg goes up, the price goes down: this indicates an inverse/negative relationship
between these two variables. Highway mpg could potentially be a predictor of price.

We can examine the correlation between 'highway-mpg' and 'price' and see it's approximately -0.704
"""
df[["highway-mpg", "price"]].corr()


""" ########################### Weak Linear Relationship
Let's see if "Peak-rpm" as a predictor variable of "price"."""
sns.regplot(x="peak-rpm", y="price", data=df)

"""
Peak rpm does not seem like a good predictor of the price at all since the regression line 
is close to horizontal. Also, the data points are very scattered and far from the fitted line, 
showing lots of variability. Therefore it's it is not a reliable variable.

We can examine the correlation between 'peak-rpm' and 'price' and see it's approximately -0.101616 """
df[['peak-rpm','price']].corr()

#The correlation is 0.0823, the non-diagonal elements of the table.
df[["stroke","price"]].corr() 
#There is a weak correlation between the variable 'stroke' and 'price.'
# as such regression will not work well.  We #can see this use "regplot" to demonstrate this.
sns.regplot(x="stroke", y="price", data=df)


"""
############################# Categorical variables
These are variables that describe a 'characteristic' of a data unit, and are selected from 
a small group of categories. The categorical variables can have the type "object" or "int64". 
A good way to visualize categorical variables is by using boxplots.

Let's look at the relationship between "body-style" and "price".
"""
sns.boxplot(x="body-style", y="price", data=df)
"""
We see that the distributions of price between the different body-style categories have a 
significant overlap, and so body-style would not be a good predictor of price. 
Let's examine engine "engine-location" and "price":
"""
sns.boxplot(x="engine-location", y="price", data=df)

"""
Here we see that the distribution of price between these two engine-location categories, 
front and rear, are distinct enough to take engine-location as a potential good predictor of price.

Let's examine "drive-wheels" and "price"."""
# drive-wheels
sns.boxplot(x="drive-wheels", y="price", data=df)

"""
Here we see that the distribution of price between the different drive-wheels categories differs; 
as such drive-wheels could potentially be a predictor of price.

################################### 3. Descriptive Statistical Analysis
Let's first take a look at the variables by utilizing a description method.

The describe function automatically computes basic statistics for all continuous variables. Any NaN values are automatically skipped in these statistics.

This will show:

# the count of that variable
# the mean
# the standard deviation (std)
# the minimum value
# the IQR (Interquartile Range: 25%, 50% and 75%)
# the maximum value
We can apply the method "describe" as follows:

"""
df.describe()
#The default setting of "describe" skips variables of type object. 
#We can apply the method "describe" on the variables of type 'object' as follows:
df.describe(include=['object'])


""" ################################# Value Counts
Value-counts is a good way of understanding how many units of each characteristic/variable we have. 
We can apply the "value_counts" method on the column 'drive-wheels'. 
Don’t forget the method "value_counts" only works on Pandas series, not Pandas Dataframes. 
As a result, we only include one bracket "df['drive-wheels']" not two brackets "df[['drive-wheels']]".
"""
df['drive-wheels'].value_counts()
#We can convert the series to a Dataframe as follows :
df['drive-wheels'].value_counts().to_frame()
# Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" 
#and rename the column 'drive-wheels' to 'value_counts'.
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts

# Now let's rename the index to 'drive-wheels':
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts

# We can repeat the above process for the variable 'engine-location'.
# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)
"""Examining the value counts of the engine location would not be a good predictor variable for the price.
 This is because we only have three cars with a rear engine and 198 with an engine in the front, 
 this result is skewed. Thus, we are not able to draw any conclusions about the engine location.
 
#################################### 4. Basics of Grouping
The "groupby" method groups data by different categories. The data is grouped based on one or 
several variables and analysis is performed on the individual groups.

For example, let's group by the variable "drive-wheels". We see that there are 3 different 
categories of drive wheels."""
df['drive-wheels'].unique()
"""If we want to know, on average, which type of drive wheel is most valuable, 
we can group "drive-wheels" and then average them.

We can select the columns 'drive-wheels', 'body-style' and 'price', 
then assign it to the variable "df_group_one"."""
df_group_one = df[['drive-wheels','body-style','price']]

# We can then calculate the average price for each of the different categories of data.
# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one










