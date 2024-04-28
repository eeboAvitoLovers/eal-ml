#!/usr/bin/env python
# coding: utf-8
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords as sw
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import pickle
from sklearn.ensemble import RandomForestClassifier

# Загрузим данные
data = pd.read_csv('raw_data.csv', encoding='UTF-8')

#print(data.head(), end='\n')

data = data['message']

pattern = '[0-9!#$%&()*+,./:;<=>?@[\]^_`{|}~—/"/\-№]+'

def updating(text):
    '''
    Обработаем данные:
    приведем к единому виду, 
    удалим числа и знаки, 
    не дающие никакой смысловой нагрузки
    '''
    text = text.lower()
    text = re.sub(pattern, ' ', text)
    return text

data = data.apply(updating)

ru_stop = sw.words('russian').append('привет')
# удалим стоп-слова для русского языка
tv_model = TfidfVectorizer(stop_words=ru_stop)
# результирующая спарсенная матрица:
result_words = tv_model.fit_transform(data)

#Теперь приступим к кластеризации
#Определим оптимальное количество кластеров с помощью ssd-метрики
ssd = []

for k in range(2, 25):
    model = KMeans(n_clusters=k)
    model.fit(result_words)

    ssd.append(model.inertia_)

plt.figure(dpi=150)
plt.plot(range(2, 25), ssd, 'o--');
plt.savefig('ssd.png')

# А также посмотрим на silhouette_score
from sklearn.metrics import silhouette_score

silhouettes = []

for k in range(2, 25):
    model = KMeans(n_clusters=k)
    model.fit(result_words)

    silhouettes.append(silhouette_score(result_words, model.labels_))
    
plt.figure(dpi=200)
plt.plot(range(2, 25), silhouettes, 'o--')
plt.savefig('silhouette.png')

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

best_kmeans_silhouette = -1
best_kmeans_calinski_harabasz = 0
best_kmeans_davies_bouldin = 1
best_n = None

#Определение количества кластеров
for n_clust in range(2, 25):
    model = KMeans(n_clusters=n_clust)
    model.fit(result_words)

    kmeans_silhouette = silhouette_score(result_words, model.labels_)
    kmeans_calinski_harabasz = calinski_harabasz_score(result_words.toarray(), model.labels_)
    kmeans_davies_bouldin = davies_bouldin_score(result_words.toarray(), model.labels_)
    
    if kmeans_silhouette > best_kmeans_silhouette:
        best_kmeans_silhouette = kmeans_silhouette
        
    if kmeans_calinski_harabasz > best_kmeans_calinski_harabasz:
        best_kmeans_calinski_harabasz = kmeans_calinski_harabasz
        best_n = n_clust
        
    if kmeans_davies_bouldin < best_kmeans_davies_bouldin:
        best_kmeans_davies_bouldin = kmeans_davies_bouldin

# Выводим результаты
print('KMeans:')
print('Best n_clusters:', best_n)
print('Silhouette score:', best_kmeans_silhouette)
print('Calinski-Harabasz index:', best_kmeans_calinski_harabasz)
print('Davies-Bouldin index:', best_kmeans_davies_bouldin)

# Посмотрим на метрику silhouette_score на другом графике
#from yellowbrick.cluster import SilhouetteVisualizer

#for k in range(2, 15):
    #model = KMeans(n_clusters = k)

    #visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    #visualizer.fit(result_words)
    #visualizer.show()

# Выберем значение 13, которое подходит по методу локтя и колена
k_means = KMeans(n_clusters=13)
# найдем предсказания
cluster_labels = k_means.fit_predict(result_words)

data = pd.DataFrame(data)
data['cluster'] = cluster_labels

from sklearn.model_selection import train_test_split
X = data['message']
y = data['cluster']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

ru_stop = sw.words('russian').append('привет') # удалим стоп-слова для русского языка
tv_model = TfidfVectorizer(stop_words=ru_stop)

tv_model.fit(X_train)
X_train_tv = tv_model.transform(X_train)
X_test_tv = tv_model.transform(X_test)

#сравним популярные модели классификации
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train_tv,y_train);

from sklearn.linear_model import LogisticRegression
log = LogisticRegression(max_iter=1000)
log.fit(X_train_tv,y_train);

from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(X_train_tv,y_train);

rfc = RandomForestClassifier(max_depth=20, n_jobs=-1)
rfc.fit(X_train_tv, y_train)


from sklearn.metrics import classification_report, confusion_matrix
def report(model):
    preds = model.predict(X_test_tv)
    print(classification_report(y_test,preds))
    confusion_matrix(y_pred=preds,y_true=y_test)

print('SVC')
report(svc)

print('LOG')
report(log)

print('NB')
report(nb)

print('RFC')
report(rfc)

#После сравнения метод случайных лесов оказался лучше

# Загрузим эту модель в файл, чтобы можно было работать с новыми данными
pipe = Pipeline([('tv_model',TfidfVectorizer(stop_words=ru_stop)),('rfc', RandomForestClassifier(max_depth=20))])

pipe.fit(data['message'], data['cluster']) # обучим итоговую модель на всех данных

with open('model.pkl', 'wb') as f:
    pickle.dump(pipe, f)
# Моделью можно пользоваться!



