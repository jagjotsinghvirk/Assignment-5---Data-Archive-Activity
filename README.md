# Assignment-5---Data-Archive-Activity
#Description
K Means Clustering
K means is an iterative clustering algorithm that aims to find local maxima in each iteration. This algorithm works 
in these 5 steps:
1. Specify the desired number of clusters K: Let us choose k=2 for these 5 data points in 2-D space.
2. Randomly assign each data point to a cluster: Let’s assign three points in cluster 1 shown using red 
color and two points in cluster 2 shown using grey color.
3. Compute cluster centroids: The centroid of data points in the red cluster is shown using red cross and 
those in grey cluster using grey cross.
4. Re-assign each point to the closest cluster centroid: Note that only the data point at the bottom is 
assigned to the red cluster even though it’s closer to the centroid of grey cluster. Thus, we assign that data 
point into grey cluster.
5. Re-compute cluster centroids: Now, re-computing the centroids for both the clusters

Prerequisites & Installing
1. Loading Libraries by importing pandas and numpy with matplot for ploting to indentify clusters
 #Load Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
2. Loading dataset by giving pd.read command and 
#Load Dataset
dataset = pd.read_csv('./Downloads/illnessstudy.csv')
dataset.head()

#Breakdown of Tests
#Determine optimum number of clusters
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10,random_state = 100)
    kmeans.fit(data_transformed)
    wcss.append(kmeans.inertia_)
    
#Plot Elbow Method
plt.plot(range(1, 10), wcss,marker='o')
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') #within cluster sum of squares
plt.show()  
![image](https://user-images.githubusercontent.com/95555472/144732544-a477d8e4-8a16-46a9-ba66-28f740ccda2b.png)

#Acknowledgement
#Plot of 2 Clusters
![image](https://user-images.githubusercontent.com/95555472/144732695-1eb5424b-0891-4a9c-b8e6-8c165e10359d.png)
Red is showing Cluster 2
Cluster 2 is more accurate and dense than Cluster 1
Cluster 2 has highest Coefficient value
