# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 10:03:37 2026

@author: shree
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\DATASETS\logit classification.csv")

x= df.iloc[:,[2,3]].values
y= df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train ,y_train)
y_pred= classifier.predict(x_test)


from sklearn.preprocessing import Normalizer
nm = Normalizer()
x_train = nm.fit_transform(x_train)
x_test = nm.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
ml= MultinomialNB()
ml.fit(x_train ,y_train)
ml_y_pred = ml.predict(x_test)


from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,ml_y_pred)
print(ac) 


from sklearn.naive_bayes import BernoulliNB
be = BernoulliNB()
be.fit(x_train ,y_train)
be_y_pred= be.predict(x_test)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,be_y_pred)
print(ac) 


'''
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test= sc.transform(x_test)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,y_pred)
'''
from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,ml_y_pred)
print(ac) 


'''
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform()         
'''