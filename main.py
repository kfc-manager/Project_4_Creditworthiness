import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import csv
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn import svm

#read flash.dat to a list of lists
datContent = [i.strip().split() for i in open("./kredit.dat").readlines()]

#write it as a new CSV file
with open("./kredit.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(datContent)

#naming the labels of the columns
columns = ['Status of existing checking account','Duration in month','Credit history','Purpose','Credit amount','Savings account/bonds','Present employment since','Installment rate in percentage of disposable income',
'Personal status and sex','Other debtors/guarantors','Present residence since','Porperty','Age in years','Other installment plans','Housing','Number of existing credits at this bank','Job','Number of people being liable to provide maintenance for',
'Telephone','Foreign worker','Creditworthy']

#creating the dataframe
df = pd.read_csv('./kredit.csv',names=columns)
df.head()

style = OneHotEncoder()

df.loc[df['Purpose'] == '?', 'Purpose'] = 'Purpose ?'
df.loc[df['Present employment since'] == '?', 'Present employment since'] = 'Present employment since ?'
df.loc[df['Job'] == '?', 'Job'] = 'Job ?'
df.loc[df['Foreign worker'] == '?', 'Foreign worker'] = 'Foreign worker ?'

non_numerics = df.select_dtypes(include='object')
non_numerics = non_numerics.drop('Telephone',axis=1) #only has two classes, which means we can transform it within the column to 0 for A191 and 1 for A192
df.loc[df['Telephone'] == 'A191', 'Telephone'] = 0.0
df.loc[df['Telephone'] == 'A192', 'Telephone'] = 1.0
df['Telephone'] = df['Telephone'].astype('int64')
for i in non_numerics.columns.tolist():
    transformation = style.fit_transform(df[[i]]) #transform column i
    df = df.join(pd.DataFrame(transformation.toarray(), columns=style.categories_[0])) #add new categories (of transformation) to our dataframe
    for col in style.categories_[0]:
        df[col] = df[col].astype('int64')
    df = df.drop(i, axis=1) #dropping old column since we transformed its information

df.loc[df['Creditworthy'] == 1, 'Creditworthy'] = 1.0
df.loc[df['Creditworthy'] == 2, 'Creditworthy'] = -1.0

df = df.astype('float64')

df.head()

def z_score_normalize(X, indices):
    for i in indices:
        column = np.array([v[i] for v in X])
        mean = np.mean(column)
        std = np.std(column)
        for j, v in enumerate(column):
            X[j][i] = (v - mean) / std
    for j in range(0, len(X)):
        X[j] = np.array(X[j], dtype='float64')
    return X

numerical_non_missing = [
    'Duration in month', 
    'Credit amount', 
    'Age in years', 
    'Installment rate in percentage of disposable income', 
    'Present residence since', 
    'Number of existing credits at this bank', 
    'Number of people being liable to provide maintenance for'
]

for v in numerical_non_missing:
    df[v] = df[v].astype('float64')

indices = [df.columns.get_loc(v) for v in numerical_non_missing]

def loss_function(y_pred, y_true):
    if y_pred == y_true:
        return 0
    if y_pred == 1:
        return 5
    return 1

def imperical_risk(y_pred, y_true):
    loss = 0
    for i in range(0, len(y_pred)):
        loss += loss_function(y_pred[i], y_true[i])
    return loss / len(y_pred)

def decision(threshold, y):
    for i, v in enumerate(y):
        if y[i] < threshold:
            y[i] = -1
        else:
            y[i] = 1
    return y

def neural_network_classifier():
    splits = 5
    kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=0)

    X = df[[i for i in df.columns.tolist() if i != 'Creditworthy']].values
    y = df['Creditworthy'].values

    params = {
        'dropout': [0.1, 0.2, 0.3, 0.5, 0.6],
        'threshold': [0.5, 0.6, 0.7, 0.8, 0.9]
    }

    test_split = [v for v in kf.split(X, y)]

    params_star = {
        'dropout': 0.1,
        'threshold': 0.5,
    }

    risk = 0
    min_risk_Si = np.inf

    for i, (rest_index, test_index) in enumerate(test_split):
        X_test = X[test_index]
        X_test = z_score_normalize(X_test, indices)
        y_test = y[test_index]

        X_rest = X[rest_index]
        y_rest = y[rest_index]

        tune_split = [v for v in kf.split(X_rest, y_rest)]
        min_risk_without_Si = np.inf
        params_star_i = {
            'dropout': 0.1,
            'threshold': 0.5,
        }

        for dropout in params['dropout']:
            for threshold in params['threshold']:

                risk_without_Si = 0

                for j, (train_index, tune_index) in enumerate(tune_split):
                    X_train = X[train_index]
                    X_train = z_score_normalize(X_train, indices)
                    y_train = y[train_index]

                    X_tune = X[tune_index]
                    X_tune = z_score_normalize(X_tune, indices)
                    y_tune = y[tune_index]

                    model_ij = tf.keras.models.Sequential([
                        tf.keras.layers.Input(shape=(64,)),
                        tf.keras.layers.Dropout(dropout),
                        tf.keras.layers.Dense(16, activation='relu'),
                        tf.keras.layers.Dropout(dropout),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                    model_ij.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
                    model_ij.fit(x=X_train, y=y_train, epochs=50)

                    y_pred = decision(threshold, model_ij.predict(X_tune))
                    risk_Sj = imperical_risk(y_pred, y_tune)
                    risk_without_Si += risk_Sj

                risk_without_Si = risk_without_Si / splits

                if min_risk_without_Si > risk_without_Si:
                    params_star_i = {
                        'dropout': dropout,
                        'threshold': threshold,
                    }
                    min_risk_without_Si = risk_without_Si
            
        X_rest = z_score_normalize(X_rest, indices)

        model_i = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(64,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(params_star_i['dropout']),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model_i.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model_i.fit(x=X_rest, y=y_rest, epochs=50)
        y_pred = decision(params_star_i['threshold'], model_i.predict(X_test))

        risk_Si = imperical_risk(y_pred, y_test)
        #print(f"Recall: {recall_score(y_test, y_pred)}")
        #print(f"Precision: {precision_score(y_test, y_pred)}")
        risk += risk_Si
        if min_risk_Si > risk_Si:
            min_risk_Si = risk_Si
            params_star = params_star_i

    risk = risk / len(test_split)

    X = z_score_normalize(X, indices)
    # TODO model = svm.SVC(C=params_star['C'], degree=params_star['degree'], kernel='poly', random_state=0, class_weight='balanced',)
    print(risk)
    print(params_star)

neural_network_classifier() 
