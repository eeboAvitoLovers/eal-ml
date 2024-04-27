#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
from sqlalchemy import create_engine, inspect

engine = create_engine("postgresql://eebo:eebo@194.87.234.96:5432/eebo")
#inspector = inspect(engine)
#print(inspector.get_table_names())

tables = ['messages', 'clusters']
messages = pd.read_sql_table(tables[0], con=engine, columns=['message'])
clusters = model.predict(messages['message'])

ready_data = pd.concat(objs=[messages, pd.DataFrame(clusters, columns=['cluster'])], axis=1)

fig = plt.figure(figsize=(5, 4), dpi=150)
sns.countplot(data=ready_data, x='cluster');
plt.savefig('distrib_by_class.png')

clusters = pd.DataFrame(clusters, columns=['clusters'])
clusters.to_sql('clusters', con=engine, if_exists='replace', index=True, index_label='ticket_id')

clust = pd.read_sql_table(tables[1], con=engine)
print(clust)
