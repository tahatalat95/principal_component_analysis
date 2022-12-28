#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# reading data set
raw_data = pd.read_csv('bodyPerformance.csv')
raw_data.head(16)


# In[3]:


# data set preprocessing 
raw_data.drop(columns=["class"], inplace=True)
raw_data.head()


# In[4]:


# first we need to make the data only numerical
# so we replace gender feacture categorical values with numerical values
# or we could simply drop it 


# In[5]:


raw_data=raw_data.replace( {'gender':{'M': 1,'F':0}} )
raw_data.head(16)


# In[6]:


# great function to normalize data 
data_norm=raw_data


# In[7]:


def normalization(data):
    for col in data.columns:
        data_norm[col]=(data[col]-data[col].mean())/data[col].std()
        
    return data_norm


# In[8]:


normalization(raw_data).head(16)


# In[9]:


# then we calculate covariance 
data_dismean=data_norm
cov=data_dismean


# In[10]:


def covariance (data):
    for col in data.columns :
        data_dismean[col]=data_dismean[col]-data_dismean[col].mean()

    cov=np.dot(data_dismean.T,data_dismean)/(len(data_dismean)-1)
    return cov


# In[11]:


coveriance_matrix = covariance (data_dismean) 


# In[12]:


# then we use ready to use function to calculate eig


# In[13]:


from numpy.linalg import eig
eigenvalues, eigenvectors=eig(coveriance_matrix)


# In[14]:


eigen_vals=pd.DataFrame({
    'eigenvalue' : eigenvalues,
})
eigen_vals


# In[15]:


import matplotlib.pyplot as plt
plt.bar(["ev_" + str(i+1) for i in range(len(eigenvalues))], eigenvalues)


# In[16]:


# the first five eigen vectors are the best to represent data


# In[17]:


best_vectors=eigenvectors[:,:5] 


# In[18]:


print(best_vectors)


# In[19]:


new_data = np.dot(data_norm.values,best_vectors)


# In[21]:


headers=['feature #1','feature #2','feature #3','feature #4','feature #5']
new_data_df = pd.DataFrame(new_data,columns=headers)


# In[22]:


new_data_df


# In[23]:


new_data_df.to_csv('new_datase.csv')

