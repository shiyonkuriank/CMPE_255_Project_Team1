#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
get_ipython().run_line_magic('run', 'movies_preprocessing.ipynb import Add')


# In[3]:


x = movies[['revenue','vote_count', 'runtimes']]


# In[4]:


kmeans = KMeans(3)
kmeans.fit(x)


# In[5]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[6]:


data_with_clusters = movies.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['revenue'],data_with_clusters['vote_count'], data_with_clusters['runtimes'],c=data_with_clusters['Clusters'],cmap='rainbow')


# In[ ]:




