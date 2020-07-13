#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#from cuml.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[ ]:


# Importing dataset and examining it
dataset = pd.read_csv("Clients.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())
print(dataset.median())


# In[ ]:


# Converting Categorical features into Numerical features
dataset['job'] = dataset['job'].map({'admin':0, 'blue-collar':1, 'entrepreneur':2, 'housemaid' :3, 'management' :4, 'retired' :5,
                 'self-employed' :6,'services' :7,'student' :8,'technician' :9,'unemployed':10})
dataset['marital'] = dataset['marital'].map({'divorced':0, 'married':1, 'single':2})
dataset['education'] = dataset['education'].map({'primary':0, 'secondary':1, 'tertiary':2})
dataset['default'] = dataset['default'].map({'no':0, 'yes':1})
dataset['housing'] = dataset['housing'].map({'no':0, 'yes':1})
dataset['personal'] = dataset['personal'].map({'no':0, 'yes':1})
dataset['term'] = dataset['term'].map({'no':0, 'yes':1})


# In[ ]:


print(dataset.info())
# Plotting Correlation Heatmap
corrs = dataset.corr()
figure = ff.create_annotated_heatmap(
    z=corrs.values,
    x=list(corrs.columns),
    y=list(corrs.index),
    annotation_text=corrs.round(2).values,
    showscale=True)
offline.plot(figure,filename='corrheatmap.html')


X=dataset
print(X.info())


# In[ ]:


#Defining if balance was high or low
def converter(column):
    if column <= 1354:
        return 0 # Low
    else:
        return 1 # High
    
X['balance'] = X['balance'].apply(converter)
print(X.head())
print(X.info())  


# In[ ]:


# Defining if age was higher or lower than mean age
def converter1(column):
    if column <= 40:
        return 0 # Low
    else:
        return 1 # High
    
X['age'] = X['age'].apply(converter1)
print(X.head())
print(X.info()) 


# In[ ]:


#Handling NA values
X = X.fillna(0)
#changing datatype of column job
X['job']=X.job.astype(int)
print(X.head())
print(X.info()) 


# In[ ]:


# Dividing data into subsets
#Personal + housing Data
subset1 = X[['housing','education','balance']]

#Loan related Data
subset2 = X[['default','personal','term']]

#Exclusive personal data
subset3= X[['age','job','marital','education']]

# Dividing data into subsets
subset4=X[['default','balance','housing','personal','term']]


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
X3 = feature_scaler.fit_transform(subset3)
X4 = feature_scaler.fit_transform(subset4)


# In[ ]:


# Analysis on subset1 - Personal Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[ ]:


# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 3)
kmeans.fit(X1)


# In[ ]:


# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =10,n_iter=3000)
x_tsne = tsne.fit_transform(X1)


# In[ ]:


housing = list(X['housing'])
#marital = list(X['marital'])
education = list(X['education'])
balance= list(X['balance'])

data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'balance: {a}, housing: {b}, education:{c}' 
                                      for a,b,c in list(zip(balance,housing,education))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE1.html')


# In[ ]:


# Analysis on subset2 - Loan Related Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[ ]:


# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 4)
kmeans.fit(X2)


# In[ ]:


# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =10,n_iter=5000)
x_tsne = tsne.fit_transform(X2)


# In[ ]:


default = list(X['default'])
housing = list(X['housing'])
personal = list(X['personal'])
#term = list(X['term'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'default:{a}, housing:{b}, personal:{c}' 
                                      for a,b,c in list(zip(default,housing,personal))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE2.html')


# In[ ]:


# Analysis on subset3 - Exclusive Personal Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X3)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[ ]:



# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X3)


# In[ ]:


# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =50,n_iter=5000)
x_tsne = tsne.fit_transform(X3)


# In[ ]:


age = list(X['age'])
job = list(X['job'])
marital = list(X['marital'])
education = list(X['education'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'age:{a}, job:{b}, marital:{c},education:{d}' 
                                      for a,b,c,d in list(zip(age,job,marital,education))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE3.html')


# In[ ]:


# Analysis on subset4 - Exclusive Personal Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X4)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[ ]:



# Running KMeans to generate labels
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X4)


# In[ ]:


# Implementing t-SNE to visualize dataset
tsne = TSNE(n_components = 2, perplexity =50,n_iter=5000)
x_tsne = tsne.fit_transform(X4)


# In[ ]:


default = list(X['default'])
balance = list(X['balance'])
housing = list(X['housing'])
personal = list(X['personal'])
term = list(X['term'])
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'default:{a}, balance:{b}, housing:{c},personal:{d},term:{e}' 
                                      for a,b,c,d,e in list(zip(default,balance,housing,personal,term))],
                                hoverinfo='text')]

layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 1000, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='t-SNE4.html')


# In[ ]:


###
### Part B code for Visualisation based on Patches dataset 
###
###


# In[36]:


## Imported the dataset for the patches dataset
patches_datavalue = pd.read_csv('E:/SEM2/PYTHON/CA02/Datasets/Patches.csv',sep=',')


# In[37]:



patches_datavalue.isnull().sum()


# Plotting Correlation Heatmap
correlation_pt = patches_datavalue.corr()
fig = ff.create_annotated_heatmap(
    z=correlation_pt.values,
    x=list(correlation_pt.columns),
    y=list(correlation_pt.index),
    annotation_text=correlation_pt.round(2).values,
    showscale=True)
offline.plot(fig,filename='corrheatmap.html')


X = patches_datavalue

### Converting the data for column tree to binomial i.e ZERO and ONE 
X['Tree'] = X['Tree'].map({'Other':1, 'Spruce':0})

### Calculating the mean for all the columns in the dataset
X.mean()


### Calucalting the median for all the columns in the dataset
X.median()


# In[38]:


# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()


# In[11]:


### Code for Subset 1 
# Defining if slope for mean values 16.50 with high or low
def converter_slope(column):
    if column <= 16.5:
        return 1 # Low
    else:
        return 0 # High
    
### Apply median for slope variable with mean using the function to classifiy the data as 1 or 0 
X['Slope'] = X['Slope'].apply(converter_slope)


## Cluster 1:  Used slope here for clustering data based on slope mean  subset 1 Good one
subset1 = X[['Slope', 'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways']]
 

X1 = feature_scaler.fit_transform(subset1)


# In[12]:


# Finding the number of clusters (K) for subset 1  - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X1)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[13]:


# Running KMeans to generate labels with number of cluster = 1
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X1)


# In[14]:


# Implementing t-SNE to visualize dataset for subset 1 
tsne = TSNE(n_components = 2, perplexity =50,n_iter=2000,random_state=1)
x_tsne = tsne.fit_transform(X1)


# In[15]:


Tree = list(X['Tree'])
Elevation = list(X['Elevation'])
Slope = list(X['Slope'])
Horizontal_Distance_To_Hydrology = list(X['Horizontal_Distance_To_Hydrology'])
Vertical_Distance_To_Hydrology = list(X['Vertical_Distance_To_Hydrology'])
Horizontal_Distance_To_Roadways = list(X['Horizontal_Distance_To_Roadways'])


# In[16]:


### plot with all the paramters 
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Horizontal_Distance_To_Hydrology: {a},Elevation: {b},Slope: {c},Tree : {d}, Vertical_Distance_To_Hydrology: {e},Horizontal_Distance_To_Roadways: {f}' for a,b,c,d,e,f in list(zip(Horizontal_Distance_To_Hydrology, Elevation, Slope,Tree , Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways))],
                                hoverinfo='text')]
layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='patches_subset_1_t-SNE.html')


# In[17]:


X = patches_datavalue

### Converting the data for column tree to binomial i.e ZERO and ONE 
X['Tree'] = X['Tree'].map({'Other':1, 'Spruce':0})


# In[27]:


### Code for Subset 2
# Defining if Horizontal_Distance_To_Roadways median values of 180 with  high or low
def converter_Horizontal_Distance_To_Hydrology(column):
    if column <= 180.0:
        return 1 # Low
    else:
        return 0 # High
    
### Converting the  using the function.
X['Horizontal_Distance_To_Hydrology'] = X['Horizontal_Distance_To_Hydrology'].apply(converter_Horizontal_Distance_To_Hydrology)

## Subset 2 Best can be displayed
subset2 = X[['Slope','Elevation', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology']]

X2 = feature_scaler.fit_transform(subset2)


# In[28]:


# Finding the number of clusters (K) for subset 1  - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[29]:


# Running KMeans to generate labels with number of cluster = 1
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X2)


# In[30]:


# Implementing t-SNE to visualize dataset for subset 1 
tsne = TSNE(n_components = 2, perplexity =50,n_iter=2000,random_state=1)
x_tsne = tsne.fit_transform(X2)


# In[31]:


### plot with all the paramters 
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Horizontal_Distance_To_Hydrology: {a},Elevation: {b},Slope: {c},Tree : {d}, Vertical_Distance_To_Hydrology: {e},Horizontal_Distance_To_Roadways: {f}' for a,b,c,d,e,f in list(zip(Horizontal_Distance_To_Hydrology, Elevation, Slope,Tree , Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways))],
                                hoverinfo='text')]
layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='Patches_subset_2_t-SNE.html')


# In[32]:


X = patches_datavalue

### Converting the data for column tree to binomial i.e ZERO and ONE 
X['Tree'] = X['Tree'].map({'Other':1, 'Spruce':0})


# In[39]:


### Code for Subset 3
# Defining if Elevation for mean value was high or low
def converter_Elevation(column):
    if column <= 2749.32:
        return 1 # Low
    else:
        return 0 # High

### Converting the  using the function.
X['Elevation'] = X['Elevation'].apply(converter_Elevation)

## Subset 3 proper cluster mean of elevation 
subset3 = X[['Slope','Elevation','Tree', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways']]

X3 = feature_scaler.fit_transform(subset3)


# In[40]:


# Finding the number of clusters (K) for subset 1  - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X3)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()


# In[41]:


# Running KMeans to generate labels with number of cluster = 1
kmeans = KMeans(n_clusters = 2)
kmeans.fit(X3)


# In[42]:


# Implementing t-SNE to visualize dataset for subset 1 
tsne = TSNE(n_components = 2, perplexity =50,n_iter=2000,random_state=1)
x_tsne = tsne.fit_transform(X3)


# In[45]:


### plot with all the paramters 
data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
                                text=[f'Horizontal_Distance_To_Hydrology: {a},Elevation: {b},Slope: {c},Tree : {d}, Vertical_Distance_To_Hydrology: {e},Horizontal_Distance_To_Roadways: {f}' for a,b,c,d,e,f in list(zip(Horizontal_Distance_To_Hydrology, Elevation, Slope,Tree , Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways))],
                                hoverinfo='text')]
layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
                    xaxis = dict(title='First Dimension'),
                    yaxis = dict(title='Second Dimension'))
fig = go.Figure(data=data, layout=layout)
offline.plot(fig,filename='Patches_subset_3_t-SNE.html')


# In[ ]:




