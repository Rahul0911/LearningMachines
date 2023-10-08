# -*- coding: utf-8 -*-

#**NASA-NEAREST EARTH OBJECTS**

'Data Source: https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects'


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('/content/neo_v2.csv')

dataset.head()

dataset.shape

dataset.isnull().sum()

dataset[dataset.columns].nunique()

#In the dataset, the columns 'ID' and 'name' are not contributing much whereas the columns 'orbiting_body' and 'sentry_object' has
#only one unique value which will induce bias in the model if we don't drop them
data = dataset.drop(columns=['id','name','orbiting_body','sentry_object'], axis=1)
data.head()

#I am encoding the target variable using the Label Encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['hazardous'] = le.fit_transform(data['hazardous'])

data.head()

sns.stripplot(data, x='hazardous', y='est_diameter_min', hue='hazardous')

sns.boxplot(data, x='hazardous', y='est_diameter_max', hue='hazardous')

sns.barplot(data, x='hazardous', y='relative_velocity', hue='hazardous')

sns.distplot(data['miss_distance'])

sns.stripplot(data, x='hazardous', y='absolute_magnitude', hue='hazardous')

#Here, I am splitting the dataset into Dependant and Indenpendent variables
X = data.drop(columns=['hazardous'], axis=1)
y = data['hazardous']

#We are splitting the dataset into train and test dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#After splitting the datasdet, I am scaling two columns using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train['relative_velocity'] = scaler.fit_transform(X_train['relative_velocity'].values.reshape(-1, 1))
X_test['relative_velocity'] = scaler.transform(X_test['relative_velocity'].values.reshape(-1, 1))

X_train['miss_distance'] = scaler.fit_transform(X_train['miss_distance'].values.reshape(-1, 1))
X_test['miss_distance'] = scaler.transform(X_test['miss_distance'].values.reshape(-1, 1))

X_train['absolute_magnitude'] = scaler.fit_transform(X_train['absolute_magnitude'].values.reshape(-1, 1))
X_test['absolute_magnitude'] = scaler.transform(X_test['absolute_magnitude'].values.reshape(-1, 1))

X_train.head()

X_train.shape

# Here I am using the Randomized Search CV to figure out the best model parameters and configuration for random forest classifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

#Creating the parameter grid, from which the Randomized Search CV will pick out the best parameter for best performance
param_grid = {'n_estimators': [10, 15, 30, 50, 100, 150],
              'max_depth': [3, 4, 5, 6, 7, 8, 9],
              'max_features': ['auto', 'sqrt', None] + list(np.arange(0.5, 1, 0.1)),
              'max_leaf_nodes': list(range(10, 51)),
              'min_samples_split': [2, 5, 10],
              'bootstrap': [True, False]}

classifier = RandomForestClassifier(random_state=1)

random_search = RandomizedSearchCV(classifier, param_grid, scoring= 'roc_auc', n_iter= 10, random_state=1)

search_fit = random_search.fit(X_train, y_train)

#The best model parameters are shown with this line of code
search_fit.best_params_

#The best random forest estimator with parameter is shown with this line of code
search_fit.best_estimator_

#Here I am trying to predict the Test dataset values based on the best estimator that we got above
best_estimator = search_fit.best_estimator_

y_pred = best_estimator.predict(X_test)

#After predicting the test dataset values, I am calculating the accuracy score of the model
from sklearn.metrics import accuracy_score

Accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ", Accuracy)