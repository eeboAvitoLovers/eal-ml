#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import numpy as np
from sqlalchemy import create_engine
from nltk.corpus import stopwords as sw
from sklearn.feature_extraction.text import CountVectorizer
import schedule
import time

engine = create_engine("postgresql://eebo:eebo@194.87.234.96:5432/eebo")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def process_new_data():
    # Получаем новые данные из базы данных
    new_data = pd.read_sql_query("SELECT m.id, m.message FROM messages m LEFT JOIN clusters c ON m.id = c.ticket_id WHERE c.cluster IS NULL", con=engine)

    # Делаем предсказание кластеров для новых данных
    clusters = pd.DataFrame(model.predict(new_data['message']), columns=['cluster'])

    # Объединяем новые данные с предсказанными кластерами
    new_data = pd.concat([new_data, clusters], axis=1)

    # Записываем кластеры в таблицу clusters
    clusters['ticket_id'] = new_data['id']
    clusters = clusters[['ticket_id', 'cluster']]
    clusters.to_sql('clusters', con=engine, if_exists='append', index=False)

    fig = plt.figure(figsize=(5, 4), dpi=150)
    sns.countplot(data=new_data, x='cluster');
    plt.savefig('distrib_by_class.png')

# Запускаем функцию process_new_data каждый час
schedule.every(1).hours.do(process_new_data)

while True:
    schedule.run_pending()
    time.sleep(60)

