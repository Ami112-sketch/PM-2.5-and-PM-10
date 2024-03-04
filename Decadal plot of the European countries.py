#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as smb


# In[2]:


df=pd.read_csv("/Users/Dell/Downloads/1970-1979.csv",index_col="Year 1970-1979")
#df.set_index("Year 1970-1979",inplace=True) or use index_col="the name for the index"
df.head()


# In[12]:


sns.scatterplot(data=df,color="purple")
plt.xlabel('Year')
plt.ylabel('particulate matter')


# In[7]:


df=pd.read_csv("/Users/Dell/Downloads/Decade.csv")
df.set_index("Decade",inplace=True)
df.head()


# In[35]:


df.shape


# In[36]:


df.describe()


# In[8]:


df=pd.read_csv("/Users/Dell/Downloads/Decade.csv")
df.set_index("Decade",inplace=True)
df.head()


# In[38]:


sns.scatterplot(data=df,color="purple")
plt.xlabel('Decade')
plt.ylabel('particulate matter')


# In[39]:


plt.figure(figsize=(12,8))
sns.lineplot(data=df)
plt.xlabel('Decade')
plt.ylabel('particulate matter')


# In[9]:


df1 = pd.read_csv('/Users/Dell/Downloads/year.csv')
df1.set_index('Year', inplace = True)
df1.head()


# In[10]:


plt.figure(figsize=(12,5))
sns.lineplot(data=df1)
plt.xlabel('Year')
plt.ylabel('particulate matter')


# In[12]:


df1 = pd.read_csv('/Users/Dell/Downloads/year.csv')
#df1.drop("Year",axis=1)
df1.set_index('Year', inplace = True)
df1.head()


# In[12]:


plt.figure(figsize=(12,5))
sns.scatterplot(data=df, markers=True)
plt.xlabel('Decade')
plt.ylabel('particulate matter')


# In[ ]:


plt.figure(figsize=(12,5))
sns.lineplot(data=df)
plt.xlabel('Year')
plt.ylabel('particulate matter')


# In[ ]:


df.value_counts()


# In[41]:


df0 = df1.loc[1970:1979]
df2 = df1.loc[1980:1989]
df3 = df1.loc[1990:1999]
df4 = df1.loc[2000:2009]
df5 = df1.loc[2010:2018]

df1.head()


# In[42]:


fig =  plt.figure()

ax0 = fig.add_subplot(2,3,1)
ax1 = fig.add_subplot(2,3,2)
ax2 = fig.add_subplot(2,3,3)
ax3 = fig.add_subplot(2,3,4)
ax4 = fig.add_subplot(2,3,5)



df0.plot(kind='line', ax=ax0, figsize=(15,7))
ax0.set_xlabel('')
ax0.set_ylabel('particulate matter')
ax0.set_title('1st Decade')

df2.plot(kind='line', ax=ax1, figsize=(15,7))
ax1.set_xlabel('')
#ax1.set_ylabel('particulate matter')
ax1.set_title('2nd Decade')

df3.plot(kind='line', ax=ax2, figsize=(15,7))
ax2.set_xlabel('Decade')
#ax2.set_ylabel('particulate matter')
ax2.set_title('3rd Decade')

df4.plot(kind='line', ax=ax3, figsize=(15,7))
ax3.set_xlabel('Decade')
ax3.set_ylabel('particulate matter')
ax3.set_title('4th Decade')

df5.plot(kind='line', ax=ax4, figsize=(15,7))
ax4.set_xlabel('Decade')
#ax4.set_ylabel('particulate matter')
ax4.set_title('5th Decade')

plt.subplots_adjust(hspace=0.4)

plt.show()


# In[50]:


df1_norm = df1.drop(['EU_pm10','EU_pm2.5'], axis = 1)
df1_norm.head()


# In[52]:


df1['pm10_norm'] = (df1['EU_pm10'] - df1['EU_pm10'].min())/(df1['EU_pm10'].max() - df1['EU_pm10'].min())
df1['pm2.5_norm'] = (df1['EU_pm2.5'] - df1['EU_pm2.5'].min())/(df1['EU_pm2.5'].max() - df1['EU_pm2.5'].min())
df1.head()


# In[54]:


df0 = df1_norm.loc[1970:1979]
df2 = df1_norm.loc[1980:1989]
df3 = df1_norm.loc[1990:1999]
df4 = df1_norm.loc[2000:2009]
df5 = df1_norm.loc[2010:2018]

df1.head()


# In[56]:


df_norm = df.drop(['EU_pm10','EU_pm2.5'], axis = 1)
df_norm.head()


# In[57]:


fig =  plt.figure()

ax0 = fig.add_subplot(2,3,1)
ax1 = fig.add_subplot(2,3,2)
ax2 = fig.add_subplot(2,3,3)
ax3 = fig.add_subplot(2,3,4)
ax4 = fig.add_subplot(2,3,5)



df0.plot(kind='line', ax=ax0, figsize=(15,7))
ax0.set_xlabel('')
ax0.set_ylabel('particulate matter')
ax0.set_title('1st Decade')

df2.plot(kind='line', ax=ax1, figsize=(15,7))
ax1.set_xlabel('')
#ax1.set_ylabel('particulate matter')
ax1.set_title('2nd Decade')

df3.plot(kind='line', ax=ax2, figsize=(15,7))
ax2.set_xlabel('Decade')
#ax2.set_ylabel('particulate matter')
ax2.set_title('3rd Decade')

df4.plot(kind='line', ax=ax3, figsize=(15,7))
ax3.set_xlabel('Decade')
ax3.set_ylabel('particulate matter')
ax3.set_title('4th Decade')

df5.plot(kind='line', ax=ax4, figsize=(15,7))
ax4.set_xlabel('Decade')
#ax4.set_ylabel('particulate matter')
ax4.set_title('5th Decade')

plt.subplots_adjust(hspace=0.4)

plt.show()


# In[13]:


import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as smb


# In[14]:


df = pd.read_csv('/Users/Dell/Downloads/year.csv')
df.set_index('Year', inplace = True)
df.head()


# In[15]:


df1 = df.loc[1970:1979]
df2 = df.loc[1980:1989]
df3 = df.loc[1990:1999]
df4 = df.loc[2000:2009]
df5 = df.loc[2010:2018]

df1.head(10)

