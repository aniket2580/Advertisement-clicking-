# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



import os
os.chdir(r'C:\Users\Aniket Kambli\Downloads\data science projects')
data=pd.read_csv('advertising.csv')
data.head()
data.dtypes
data['Timestamp']=pd.to_datetime(data['Timestamp'])

data['Hour']=data['Timestamp'].dt.hour
data['minute']=data['Timestamp'].dt.minute
data['time_accessed']=data['Hour']*60+data['minute']

data.columns

data['City'].nunique()
data['Country'].nunique()

X=data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage','Male', 'Hour', 'minute']]

y=data['Clicked on Ad']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)


from sklearn.linear_model import LogisticRegression

model=LogisticRegression()

model.fit(X_train,y_train)

ypred=model.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,ypred))

print(classification_report(y_test,ypred))



import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
