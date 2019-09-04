#!/usr/bin/env python
# coding: utf-8

# # Game Sales Prediction

# Numpy — It’s a built in python library which helps in doing mathematical functions such as matrix multiplication , conversion , etc
# 
# Pandas — The most important library , this library is used to import dataset and create data frames. which can be further used for analysis or prediction whatever you want to do!
# 
# Matplotlib — it’s a tool used for data visualization and representation.
# 
# %matplotlib inline — Since , I’m going to use jupyter(Ipython notebook) notebook i want my output (graph) to be inside the notebook.

# In[49]:


import pandas as pd
import pandas_profiling
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# So we are now ready , we import the data by using the above command. pd is short for pandas as mentioned above (import pandas as pd). read_csv is a function inside pandas library. train.csv is a file present in on the anaconda directory.

# In[6]:


train = pd.read_csv('Datasets/videogamesales/vgsales.csv')


# Pandas uses head function to give us an overview about the top part of the data frame. By default head() would return us the first 5 rows of the data frame. Similarly if we put head(20) it would return us the first 20 rows! interesting isn’t it ?
# Similarly we can use tail() to see the last 5 (by default) of the data frame.

# In[7]:


train.head()


# When importing a new data set for the very first time, the first thing to do is to get an understanding of the data. This includes steps like determining the range of specific predictors, identifying each predictor’s data type, as well as computing the number or percentage of missing values for each predictor.
# 
# Since few datasets are huge and would be real pain to work with we need to find useful information and eliminate unwanted information sounds easy but identifying those could be difficult. It’s said to be a good practice to look into the shape of the dataset.

# In[8]:


print(train.shape)
print(train.size)


# Let’s see some basic statistics about the data we have right now. Here the statistics of data such as count , mean , percentile ,standard deviation are seen these are important when we play with some financial data or performance related data.

# In[9]:


train.describe()


# While the output above contains lots of information, it does not tell you everything you might be interested in. For instance, you could assume that the data frame has 891 rows. If you wanted to check, you would have to add another line of code to determine the length of the data frame. While these computations are not very expensive, repeating them over and over again does take up time you could probably better use while cleaning the data.
# 
# Exploratory Data Analysis (EDA) plays a very important role in understanding the dataset. Whether you are going to build a Machine Learning Model or if it's just an exercise to bring out insights from the given data, EDA is the primary task to perform. While it's undeniable that EDA is very important, The task of performing Exploratory Data Analysis grows in parallel with the number of columns your dataset has got.
# 
# For ex: Assume you've got a dataset with 10 rows x 2 columns. It's very simply to specify those two column names separately and plot all the required plots to perform EDA. Alternatively, If the dataset has got 20 columns, you've to repeat the same above exercise for another 10 times. Now, there's another layer of complexity because the visualization that you choose for a continuous variable and categorical variable is different, hence the type of the plot changes when the data type changes.
# 
# Given all these conditions, EDA sometimes becomes a tedious task - but remember it's all driven by a set of rules - like plot boxplot and histogram for a continous variable, Measure missing values, Calculate frequency if it's categorical variable - thus giving us opportunity to automate things. That's the base of this python module pandas_profiling that helps one in automating the first-level of EDA.
# 
# For each column the following statistics - if relevant for the column type - are presented in an interactive HTML report:
# 
# Essentials: type, unique values, missing values.
# Quantile statistics: minimum value, Q1, median, Q3, maximum, range, interquartile range.
# Descriptive statistics: mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness.
# Most frequent values
# Histogram
# Correlations highlighting of highly correlated variables, Spearman and Pearson matrixes

# In[50]:


pandas_profiling.ProfileReport(train)


# Running this single line of code will create an HTML EDA report of your data. The code displayed above will create an inline output of the result; however, you could also choose to save your EDA report as an HTML file to be able to share it more easily.
# 
# The first part of the HTML EDA report will contain an overview section providing you with basic information (number of observations, number of variables, etc.). It will also output a list of warnings telling you where to double-check the data and potentially focus your cleaning efforts on.

# It is very important for a data scientist to know which representation to be used , we are first going to talk about that and then move into the code. When we visualize data we must have few things in our mind that is: How many variables are to be shown on single chart? Are there several items for a single point of data or many? Are we displaying the data for a period of time ? or are we grouping it?

# In[22]:


plt.figure(figsize = (16,10))
train['Year'].hist(bins = 40)
plt.title('Distribuição ao longo dos anos')
plt.xlabel('Anos')
plt.ylabel('Lançamentos')
plt.show()


# In[33]:


plt.figure(figsize = (16,10))
graphv1 = train.groupby(["Year"]).sum()
print(graphv1.head())
plt.plot(graphv1.index, graphv1.Global_Sales)
plt.plot(graphv1.index, graphv1.Other_Sales)
plt.plot(graphv1.index, graphv1.EU_Sales)
plt.plot(graphv1.index, graphv1.JP_Sales)
plt.plot(graphv1.index, graphv1.NA_Sales)
plt.title('Valor total de jogos vendidos por ano')
plt.xlabel('Ano')
plt.ylabel('Valor vendido')
plt.show()


# In[47]:


plt.figure(figsize = (16,10))

plt.subplot (1,2,1)
graphv1 = train.groupby(["Year"]).sum()
print(graphv1.head())
plt.plot(graphv1.index, graphv1.Global_Sales)
plt.title('Valor total de jogos vendidos por ano no mundo')
plt.xlabel('Ano')
plt.ylabel('Valor vendido')

plt.subplot(1, 2, 2)
plt.plot(graphv1.index, graphv1.NA_Sales, 'k-')
plt.title('Valor total de jogos vendidos por ano na América do Norte')
plt.xlabel('Ano')
plt.ylabel('Valor vendido')

plt.show()


# In[ ]:




