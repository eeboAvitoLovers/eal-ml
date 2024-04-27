#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sqlalchemy import create_engine, inspect

engine = create_engine("postgresql://eebo:eebo@194.87.234.96:5432/eebo")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

tables = ['messages', 'clusters']
messages = pd.read_sql_table(tables[0], con=engine, columns=['id', 'message'])
clusters = pd.DataFrame(model.predict(messages['message']), columns=['cluster'])

ready_data = pd.concat(objs=[messages, clusters], axis=1)

fig = plt.figure(figsize=(5, 4), dpi=150)
sns.countplot(data=ready_data, x='cluster');
plt.savefig('distrib_by_class.png')

clusters['ticket_id'] = messages['id']
clusters = clusters[['ticket_id', 'cluster']]
clusters.to_sql('clusters', con=engine, if_exists='replace', index=False)

clust = pd.read_sql_table(tables[1], con=engine)
print(clust)
