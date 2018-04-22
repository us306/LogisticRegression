# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 21:13:51 2018

@author: Utkarsh
"""

import pandas as pd
import numpy as np

df = pd.read_csv("DelayedFlights.csv")

df = df.drop("Unnamed: 0",1)
df = df[df["Month"].isin([10,11,12])]
df.head()

cancelled = df[df['Cancelled']==1]

cancelled.tail()

import matplotlib.pyplot as plt

font = {'size'   : 16}
plt.rc('font', **font)

#percentagecancellationswise
days_cancelled = cancelled['Cancelled'].groupby(df['DayOfWeek']).count()
days_total = df['Cancelled'].groupby(df['DayOfWeek']).count()
days_frac = np.divide(days_cancelled, days_total)
x=days_frac.index.values
week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig, ax = plt.subplots(figsize = (12,6))
ax.bar(x,days_frac*100, align='center')
ax.set_ylabel('Percentage of Flights Cancelled')
ax.set_xticks(x)
ax.set_xticklabels(week, rotation = 45)

plt.show()

#Departuretimewise
df['CRSDepTime'].head(10)
fig, ax = plt.subplots(figsize = (12,6))

ax.hist([df['CRSDepTime'], cancelled['CRSDepTime']], normed=1, bins=20, label=['All', 'Cancelled'])

ax.set_xlim(0,2400)

ax.set_xlabel('Scheduled Departure Time')
ax.set_title('Normalized histogram of Scheduled Departure Times')

plt.legend()
plt.show()

#Dayofmonth
df['DayofMonth'].head(10)

fig, ax = plt.subplots(figsize = (12,6))

ax.hist([df['DayofMonth'], cancelled['DayofMonth']], normed=1, bins=31, label=['All', 'Cancelled'])

ax.set_xlim(0,31)

ax.set_xlabel('Day of Month')
ax.set_title('Normalized histogram of Day of Month')

plt.legend()
plt.show()

#Monthwise
fig, ax = plt.subplots(figsize = (12,6))

ax.hist([df['Month'], cancelled['Month']], normed=1, bins=3, label=['All', 'Cancelled'])

ax.set_xlim(10,12)

ax.set_xlabel('Month')
ax.set_title('Normalized histogram of Months')

plt.legend()
plt.show()

#Distancewise
fig, ax = plt.subplots(figsize = (12,6))

ax.hist([df['Distance'], cancelled['Distance']], normed=1, bins=20, label=['All', 'Cancelled'])

ax.set_xlim(0,3000)
ax.set_xlabel('Flight Distance in miles')
ax.set_title('Normalized histogram of Flight Distances')

plt.legend()
plt.show()

data = df.loc[:,['DepTime','CRSDepTime','Distance','Cancelled']].values
data = pd.DataFrame(data)
data.dropna()
features = data.iloc[:,0:-1].values
labels = data.iloc[:,-1].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
f_train,f_test,l_train,l_test = train_test_split(features,labels,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(f_train,l_train)
l_pred = lr.predict(f_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(l_test,l_pred)

# tp/tp+fp
