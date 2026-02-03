import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"C:\FSDS\NIT\FSDS\4. JANUARY MONTH\8 jan decision tree\9th- DECISSION TREE\5. DECESSION TREE CODE\Social_Network_Ads.csv")


from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
dataset['Gender'] = le.fit_transform(dataset['Gender'])



x=dataset.iloc[:,0:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.tree import DecisionTreeClassifier
de_cls= DecisionTreeClassifier()
de_cls.fit(x,y)
de_cls_pred = de_cls.predict(x_test)

from sklearn.metrics import accuracy_score,root_mean_squared_error
ac=accuracy_score(y_test,de_cls_pred)
print(ac)

