#!/usr/bin/env python
# coding: utf-8

# ## SPARKS FOUNDATION¶

# ### Name: Vidhita Sabale

# ### TASK_2

# ### Problem Statment:From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.

# ## Prediction using Unsupervised ML

# #### Dataset : https://bit.ly/3kXTdox

# ## Importing the libraries

# Load the necessary python libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')


# ## Load the iris dataset

# In[3]:


df = pd.read_csv("iris.csv")


# In[4]:


df


# In[5]:


df.drop(columns = "Id" , inplace = True)
df.head(5)


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


x_df = df
x = df.iloc[:,0:4].values
x


# In[9]:


x_df.plot(kind = "line")


# In[10]:


sns.pairplot(data = x_df)


# ## Finding the optimum number of clusters for k-means classification

# In[17]:


import warnings
warnings.filterwarnings('ignore')


# In[18]:


from sklearn.cluster import KMeans
wcss=[]  ## WCSS is the sum of squares of the distances of each data point in all clusters to their respective centroids

for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++",max_iter=300,n_init=10,random_state=0)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)
wcss

#Elbow Method- to find value of k
plt.plot(range(1,11),wcss , marker='o' ,  markerfacecolor='black')
plt.title('Elbow Method')
plt.xlabel('no of clusters')
labels = ["Number Of Clusters" , "Wcss"]
plt.ylabel('wcss')  # Within cluster sum of squares   #wcss is low for higher no. of clusters
plt.legend(labels=labels)
plt.show()


# We can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.
# 
# From this we choose the number of clusters as 3.

# ## Clustering

# ### Applying kmeans to the dataset / Creating the kmeans classifier

# In[12]:


kmeans=KMeans(n_clusters=3,init="k-means++",max_iter=300,n_init=10,random_state=0)
identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[13]:


kmeans.cluster_centers_


# ## Visualising the clusters - On the first two columns

# In[16]:


plt.scatter(x[identified_clusters  == 0, 0], x[identified_clusters  == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[identified_clusters  == 1, 0], x[identified_clusters  == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[identified_clusters  == 2, 0], x[identified_clusters  == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# This concludes the K-Means Workshop.
