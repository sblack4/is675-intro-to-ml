#!/usr/bin/env python
# coding: utf-8

# IS 675: Data Mining
# Summer 2019
# ## Homework Assignment 7 (Due July, 2182019, midnight EST)
# 
# 
# 
# 1. Import appropriate libraries to cluster iris data using k-means. (10 points)
# 2. Load the iris data: first 4 attributes into X and the class attribute into y. (10 points)
# 3. Create 3 clusters using K-means algorithm. (10 points)
# 4. Save the centroids of each cluster into a variable. (10 points)
# 5. Plot (2D) the data points with color-coded clusters along with the centroids using any two features. Which two features show the best cluster separation? (30 points)
# 6. Calculate the silhouette coefficients. (10 points)
# 7. Assuming that the data is ordered by class labels, print the average silhouette coefficient for each class. (20 points)
# 


# ## 1. Import appropriate libraries to cluster iris data using k-means. (10 points)

# In[16]:


import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics.cluster import silhouette_score,silhouette_samples
import matplotlib.pyplot as plt


# ## 2. Load the iris data: first 4 attributes into X and the class attribute into y. (10 points)

# In[3]:


iris_dataset = datasets.load_iris()

X = iris_dataset.data[:,:4]
y = iris_dataset.target
print('Data Shape: ', X.shape, ' Target Shape: ', y.shape)


# ## 3. Create 3 clusters using K-means algorithm. (10 points)

# In[4]:


kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(X, y)


# ## 4. Save the centroids of each cluster into a variable. (10 points)

# In[13]:


centroids = kmeans_model.cluster_centers_
labels = kmeans_model.labels_
centroids


# ## 5. Plot (2D) the data points with color-coded clusters along with the centroids using any two features. Which two features show the best cluster separation? (30 points)

# In[19]:


pairs = itertools.combinations(centroids, 2)
pairs.


# In[ ]:


def plot__kmeans_scatter_in_2d():
    
colors = ["r.","g.","b.","c.","y.","k."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 7)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', color='k', s=50, linewidth=5)
plt.show()


# ## 6. Calculate the silhouette coefficients. (10 points)

# In[9]:


silhouette_score(X, kmeans_model.labels_, metric='sqeuclidean')


# ## 7. Assuming that the data is ordered by class labels, print the average silhouette coefficient for each class. (20 points)

# In[ ]:





# In[ ]:




