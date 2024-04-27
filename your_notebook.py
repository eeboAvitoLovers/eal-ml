#!/usr/bin/env python
# coding: utf-8

# # Конечный результат

# In[645]:


# импорт библиотек
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords as sw
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sqlalchemy import create_engine
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


engine = create_engine("dialect+driver://username:password@host:port/database")

new = pd.read_sql(name=, con=engine)
data = new['message']

# In[650]:


preds = model.predict(data)

# In[653]:


new['cluster'] = preds

# In[658]:


# Итоговое распределение по кластерам
distribution = new.groupby('cluster', as_index=False).count() #Нужно загрузить new
sns.histplot(data=distribution, bins=13, x='cluster', y='message');
plt.title('Распределение по кластерам');
