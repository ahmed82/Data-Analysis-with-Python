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
    
Identify and handle missing values
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

