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

# for example, we can calculate the correlation between variables  of type "int64" or "float64" using the method "corr":
df.corr()
# Find the correlation between the following columns: bore, stroke,compression-ratio , and horsepower.
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()  
























