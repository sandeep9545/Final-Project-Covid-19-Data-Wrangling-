#!/usr/bin/env python
# coding: utf-8

# # 1. Import the dataset using Pandas from mentioned url.

# In[1]:


import pandas as pd
import numpy as np
url = 'https://raw.githubusercontent.com/SR1608/Datasets/main/covid-data.csv'
data=pd.read_csv(url)


# In[2]:


data.head()


# # 2a. High Level Data Understanding: Find no. of rows & columns in the dataset
#  

# In[3]:


print('Number of rows in the data : ',data.shape[0])
print('Number of columns in the data : ',data.shape[1])


# # 2b. High Level Data Understanding: Data types of columns.

# In[4]:


data.dtypes


# # 2c. High Level Data Understanding: Info & describe of data in dataframe.

# In[5]:


data.info()


# In[6]:


describe=pd.DataFrame(data.describe(include='all'))
describe


# # 3a. Low Level Data Understanding : Find count of unique values in location column.

# In[7]:


print("Count of unique values in location column : ",data['location'].nunique())


# # 3b. Low Level Data Understanding : Find which continent has maximum frequency using values counts.

# In[8]:


print("Continent having maximmum frequency : ",data['continent'].value_counts().idxmax()," --> ",data['continent'].value_counts()[0])


# # 3c. Low Level Data Understanding : Find maximum & mean value in 'total_cases'.

# In[9]:


import numpy as np


# In[10]:


print("Maximum value of total cases : ",np.nanmax(data['total_cases']))
print("Minimum value of total cases : ",np.nanmin(data['total_cases']))


# # 3d. Low Level Data Understanding :Find 25%,50% & 75% quartile value in 'total_deaths'.

# In[11]:


print('25% quartile value in total_deaths : ',data['total_deaths'].quantile([.25])[0.25])
print('50% quartile value in total_deaths : ',data['total_deaths'].quantile([.50])[0.50])
print('75% quartile value in total_deaths : ',data['total_deaths'].quantile([.75])[0.75])


# # 3e. Low Level Data Understanding : Find which continent has maximum 'human_development_index'.

# In[12]:


print("Continent having maximum 'human_development_index' --> ",max(pd.DataFrame(data.groupby(['continent']).sum()['human_development_index'])['human_development_index'].index))


# # 3f. Low Level Data Understanding : Find which continent has minimum 'gdp_per_capita'.

# In[13]:


print("Continent having minimum 'human_development_index' --> ",min(pd.DataFrame(data.groupby(['continent']).sum()['gdp_per_capita'])['gdp_per_capita'].index))


# # 4. Filter the dataframe with only this columns ['continent','location','date','total_cases','total_deaths','gdp_per_capita',' human_development_index'] and update the data frame.

# In[14]:


data=data[['continent','location','date','total_cases','total_deaths','gdp_per_capita','human_development_index']]


# In[15]:


data


# # 5a. Data Cleaning  Remove all duplicates observations

# In[16]:


data=data.drop_duplicates()
data.reset_index(inplace=True,drop=True)
data


# # 5b. Data Cleaning Find missing values in all columns

# In[17]:


data.isna().sum()


# # 5c. Data Cleaning :: Remove all observations where continent column value is missing Tip : using subset parameter in dropna

# In[18]:


data=data.dropna(subset=['continent'])
data.reset_index(inplace=True,drop=True)
data


# # 5d. Data Cleaning :: Fill all missing values with 0

# In[19]:


data=data.fillna(0)
data


# # 6a. Date time format :: Convert date column in datetime format using pandas.to_datetime

# In[20]:


data['date']=pd.to_datetime(data['date'])
data.dtypes


# # 6b. Date time format ::  Create new column month after extracting month data from date column.

# In[21]:


data['month'] = data['date'].dt.month_name()


# In[22]:


data


# In[23]:


data['continents']=data['continent']


# In[24]:


data


# # 7. Data Aggregation:                                                                                                a) Find max value in all columns using groupby function on 'continent' column Tip: use reset_index() after applying groupby                                                b)Store the result in a new dataframe named 'df_groupby'. (Use df_groupby dataframe for all further analysis)

# In[25]:


df_groupby = data.groupby('continent').max()
df_groupby.reset_index(inplace=True,drop=True)
df_groupby


# # 8. Feature Engineering : Create a new feature 'total_deaths_to_total_cases' by ratio of 'total_deaths' column to 'total_cases'

# In[26]:


df_groupby['total_deaths_to_total_cases'] = df_groupby['total_deaths']/df_groupby['total_cases']


# In[27]:


df_groupby


# # 9a. Data Visualization : Perform Univariate analysis on 'gdp_per_capita' column by plotting histogram using seaborn dist plot.

# In[28]:


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# In[29]:


import seaborn as sns
sns.distplot(df_groupby['gdp_per_capita'],bins=8,kde=False)
plt.title("Histogram of gdp_per_capita")
plt.show()


# # 9b. Data Visualization : Plot a scatter plot of 'total_cases' & 'gdp_per_capita'

# In[30]:


sns.scatterplot(data=df_groupby, x="total_cases", y="gdp_per_capita")
plt.ticklabel_format(style='plain', axis='x')
plt.title('Scatterplot of total_cases and gdp_per_capita')
plt.show()


# # 9c. Data Visualization :. Plot Pairplot on df_groupby dataset.

# In[31]:


sns.pairplot(df_groupby)
plt.show()


# # 9d. Data Visualization :. Plot a bar plot of 'continent' column with 'total_cases' .

# In[34]:


sns.catplot(data=df_groupby,x='continents',y='total_cases',kind='bar',aspect=2.5)
plt.ticklabel_format(style='plain', axis='y')
plt.title('Barplot of continent column with total_cases')
plt.show()


# # 10.Save the df_groupby dataframe in your local drive using pandas.to_csv function .

# In[35]:


df_groupby.to_csv('df_groupby.csv')


# In[ ]:




