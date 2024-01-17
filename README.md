# Clustering_-crime.data-
#For data science students a clustering_crime example.
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Import Dataset
crime=pd.read_csv('/content/crime_data.csv')
crime

crime.info()

crime.drop(['Unnamed: 0'],axis=1,inplace=True)
crime

# Normalize heterogenous numerical data using standard scalar fit transform to dataset
crime_norm=StandardScaler().fit_transform(crime)
crime_norm

# DBSCAN Clustering
dbscan=DBSCAN(eps=1,min_samples=4)
dbscan.fit(crime_norm)

#Noisy samples are given the label -1.
dbscan.labels_

# Adding clusters to dataset
crime['clusters']=dbscan.labels_
crime

crime.groupby('clusters').agg(['mean']).reset_index()

# Plot Clusters
plt.figure(figsize=(10, 7))
plt.scatter(crime['clusters'],crime['UrbanPop'], c=dbscan.labels_)

# create dendrogram
fig = plt.figure(figsize=(15,8))
dendrogram = sch.dendrogram(sch.linkage(crime, method='ward'))

# Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'average')

# save clusters for chart
y_hc = hc.fit_predict(crime)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])
Clusters

crime1 = pd.concat([crime,Clusters], axis=1)
crime1

crime1.sort_values("Clusters")

kmeans = KMeans(n_clusters=14,random_state=0)
kmeans.fit(crime)

kmeans.inertia_

wcss = []
for i in range(1, 19):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(crime)
    wcss.append(kmeans.inertia_)

wcss

plt.plot(range(1, 19), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Build Cluster algorithm
from sklearn.cluster import KMeans
clusters_new = KMeans(7, random_state=0)
clusters_new.fit(crime)

clusters_new.labels_

# Converting array to dataframe
df = pd.DataFrame(clusters_new.labels_, columns =['Cluster ID'])

crime2 = pd.concat([crime,df], axis=1)
crime2

crime3=crime2.drop(['Murder'], axis=1)

crime3.groupby('Cluster ID').agg(['mean']).reset_index()

crime3['Cluster ID'].value_counts()

crime.head(4)

stscaler = StandardScaler().fit(a)
X1 = stscaler.transform(a)

dbscan = DBSCAN(eps=0.82, min_samples=6)
dbscan.fit(X1)

#Noisy samples are given the label -1.
dbscan.labels_

cl=pd.DataFrame(dbscan.labels_,columns=['Cluster'])

crime4 = pd.concat([crime,cl],axis=1)
crime4

crime4['Cluster'].value_counts()

c = pd.read_csv("crime_data.csv")
c.head()

crime=c.rename({'Unnamed: 0':'States'},axis=1)
crime.tail()

#crime.States.value_counts()
crime.info()

crime.isna().sum()

n = MinMaxScaler()
data= n.fit_transform(crime.iloc[:,1:].to_numpy())
crimes = pd.DataFrame(data, columns = crime.columns[1:])
crimes.head(4)

# create dendrogram
fig = plt.figure(figsize=(15,8))
dendrogram = sch.dendrogram(sch.linkage(crimes, method='average'))

# create dendrogram
fig = plt.figure(figsize=(15,8))
dendrogram = sch.dendrogram(sch.linkage(crimes, method='ward'))

# create clusters
hc1 = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'average')

# save clusters for chart
y2 = hc1.fit_predict(crimes)
cc = pd.DataFrame(y2,columns=['Clusters'])
cc.head(6)

crimes2 = pd.concat([crime,cc],axis=1)
crimes2.head(7)

crimes2.sort_values("Clusters").reset_index()

crimes2['Clusters'].value_counts()

kmeans = KMeans(n_clusters=4,random_state=0)
kmeans.fit(crimes)

kmeans.inertia_

wcss = []
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(crimes)
    wcss.append(kmeans.inertia_)

wcss

plt.plot(range(1, 8), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Build Cluster algorithm
cc2 = KMeans(4, random_state=8)
cc2.fit(crimes)

cc2.labels_

# Converting array to dataframe
df2 = pd.DataFrame(cc2.labels_, columns =['clusters'])

crimes3 = pd.concat([crime,df2], axis=1)
crimes3.head(4)

crimes3['clusters'].value_counts()

crimes3.groupby('clusters').agg(['mean'])

array = crimes.values
#array

stscaler = StandardScaler().fit(array)
X2 = stscaler.transform(array)

X2

dbscan2 = DBSCAN(eps=0.98, min_samples=3)
dbscan2.fit(X2)

#Noisy samples are given the label -1.
dbscan2.labels_

c2 = pd.DataFrame(dbscan2.labels_,columns=['Cluster ID'])
c2.value_counts()

crimes4 = pd.concat([crime,c2],axis=1)
crimes

