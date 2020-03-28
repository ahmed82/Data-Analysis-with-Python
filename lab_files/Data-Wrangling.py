# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 00:16:53 2020

@author: 1426391
#################   Data Wrangling

"""

"""
Data Wrangling is the process of converting data from the initial format to a format that may be better for analysis.
"""

import pandas as pd
import matplotlib.pylab as plt

filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
#Use the Pandas method read_csv() to load the data from the web address. Set the parameter  "names" equal to the Python list "headers".
df = pd.read_csv(filename, names = headers)

# To see what the data set looks like, we'll use the head() method.
df.head()

"""
As we can see, several question marks appeared in the dataframe; those are missing values which may hinder our further analysis.

So, how do we identify all those missing values and deal with them?
How to work with missing data?

Steps for working with missing data:

    1. dentify missing data
    2. deal with missing data
    3. correct data format
    
#############   Identify and handle missing values ###############
                    Identify missing values
Convert "?" to NaN
In the car dataset, missing data comes with the question mark "?". We replace "?" with NaN (Not a Number), 
which is Python's default missing value marker, for reasons of computational speed and convenience. 
Here we use the function:
    .replace(A, B, inplace = True) 
to replace A by B
"""
import numpy as np

# replace "?" to NaN
df.replace("?", np.nan, inplace = True)
df.head(5)


"""
#############   Evaluating for Missing Data ################
The missing values are converted to Python's default. We use Python's built-in functions to identify these missing values. There are two methods to detect missing data:

    .isnull()
    .notnull()
The output is a boolean value indicating whether the value that is passed into the argument is in fact missing data.
"""
missing_data = df.isnull()
missing_data.head(5)
# "True" stands for missing value, while "False" stands for not missing value.

"""
#############   Count missing values in each column 
Using a for loop in Python, we can quickly figure out the number of missing values in each column. 
As mentioned above, "True" represents a missing value, "False" means the value is present in the dataset.
 In the body of the for loop the method ".value_counts()" counts the number of "True" values.
"""
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")
""""
Deal with missing data
How to deal with missing data?
drop data
    a. drop the whole row
    b. drop the whole column
replace data
    a. replace it by mean
    b. replace it by frequency
    c. replace it based on other functions
Whole columns should be dropped only if most entries in the column are empty. In our dataset, none of the columns are empty enough to drop entirely. We have some freedom in choosing which method to replace data; however, some methods may seem more reasonable than others. We will apply each method to many different columns:

        Replace by mean:

"normalized-losses": 41 missing data, replace them with mean
"stroke": 4 missing data, replace them with mean
"bore": 4 missing data, replace them with mean
"horsepower": 2 missing data, replace them with mean
"peak-rpm": 2 missing data, replace them with mean
        Replace by frequency:

"num-of-doors": 2 missing data, replace them with "four".
Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur
        Drop the whole row:

"price": 4 missing data, simply delete the whole row
Reason: price is what we want to predict. Any data entry without price data cannot be used for prediction; therefore any row now without price data is not useful to us
Calculate the average of the column 
""""


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)
# Replace "NaN" by mean value in "normalized-losses" column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)

avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)
df["bore"].replace(np.nan, avg_bore, inplace=True)
#According to the example above, replace NaN in "stroke" column by mean.
avg_stroke=df['stroke'].astype('float').mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"].replace(np.nan, avg_stroke, inplace=True)

avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)

avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)

#   To see which values are present in a particular column, we can use the ".value_counts()" method:
df['num-of-doors'].value_counts()
# We can see that four doors are the most common type. We can also use the ".idxmax()" method to calculate for us the most common type automatically:
df['num-of-doors'].value_counts().idxmax()
#replace the missing 'num-of-doors' values by the most frequent 
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)

#Good! Now, we obtain the dataset with no missing values.

"""
#################   Correct data format
The last step in data cleaning is checking and making sure that all data is in the correct format (int, float, text or other).
In Pandas, we use
    .dtype() to check the data type
    .astype() to change the data type
"""
df.dtypes

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

"""#################   Data Standardization"""
#The formula for unit conversion is
# L/100km = 235 / mpg
# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]
# transform mpg to L/100km by mathematical operation (235 divided by mpg)
df["highway-mpg"] = 235/df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
df.rename(columns={'highway-mpg':'highway-L/100km'}, inplace=True)
# check your transformed data 
df[["city-L/100km","highway-L/100km"]].head()

"""

Consider the column of the dataframe df['a']. The colunm has been standardized.
 What is the standard deviation of the values, i.e the result of applying the following operation df['a'].std() :
Answer = 1     
"""
test_ahmed = df['highway-L/100km'].std()



"""################     Data Normalization"""
# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max() 
# show the scaled columns
df[["length","width","height"]].head()

"""################  Binning"""

df["horsepower"]=df["horsepower"].astype(int, copy=True)
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins
group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)

# Lets see the number of vehicles in each bin.
df["horsepower-binned"].value_counts()

# Lets plot the distribution of each bin.







