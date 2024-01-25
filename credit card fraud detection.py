#!/usr/bin/env python
# coding: utf-8

# In[8]:


# importing libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


data = pd.read_csv("creditcard.csv")


# In[10]:


data.head()


# In[11]:


# statistical info
data.describe()


# In[12]:


# datatype info
data.info()


# In[13]:


# check for null values
data.isnull().sum()


# In[14]:


sns.countplot(data['Class'])


# In[15]:


data_temp = data.drop(columns=['Time','Amount','Class'], axis=1)

# create dist plots
fig, ax = plt.subplots(ncols=4, nrows=7, figsize=(20, 50))
index = 0
ax = ax.flatten()

for col in data_temp.columns:
    sns.distplot(data_temp[col], ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5,w_pad=0.5,h_pad=5)


# In[16]:


sns.distplot(data['Time'])


# In[17]:


sns.distplot(data['Amount'])


# In[18]:


corr = data.corr()
plt.figure(figsize=(30,40))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[19]:


x = data.drop(columns=['Class'], axis=1)
y = data['Class']


# In[20]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaler = sc.fit_transform(x)


# In[21]:


x_scaler[-1]


# In[22]:


# train test split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.25, random_state=42, stratify=y)


# In[23]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# training
model.fit(x_train, y_train)
# tesing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# In[34]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
# training
model.fit(x_train, y_train)
# tesing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# In[43]:


from xgboost import XGBClassifier
model = XGBClassifier(n_jobs=-1)
# training
model.fit(x_train, y_train)
# tesing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# In[25]:


sns.countplot(y_train)


# In[26]:


get_ipython().system('pip install imblearn')


# In[27]:


pip install -U imbalanced-learn


# In[29]:


# hint - use combination of over sampling and under sampling
# balance the class with equal distribution
from imblearn.over_sampling import SMOTE
over_sample = SMOTE()
x_smote, y_smote = over_sample.fit_resample(x_train, y_train)


# In[30]:


sns.countplot(y_smote)


# In[35]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# training
model.fit(x_smote, y_smote)
# tesing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# In[36]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1)
# training
model.fit(x_smote, y_smote)
# tesing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# In[39]:


pip install -U xgboost


# In[42]:


from xgboost import XGBClassifier
model = XGBClassifier(n_jobs=-1)
# training
model.fit(x_smote, y_smote)
# tesing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# In[ ]:




