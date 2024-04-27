#!/usr/bin/env python
# coding: utf-8
from fastapi import FastAPI
from pydantic import BaseModel
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

table = 'messages'
messages = pd.read_sql_table(table, con=engine, columns=['message'])
preds = model.predict(messages['message'])

ready_data = pd.concat(objs=[messages, pd.DataFrame(preds, columns=['cluster'])], axis=1)
ready_data[ready_data['cluster'] == preds[-1]]

fig = plt.figure(figsize=(5, 4), dpi=150)
sns.countplot(data=ready_data, x='cluster');
plt.savefig('distrib_by_class.png')

