import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv()
x=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.25,random_state=100)
'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)
'''
from sklearn.linear_model import LogisticRegression
classsifier= LogisticRegression()
classsifier.fit(x_train,y_train)
y_pred= classsifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,y_pred)
print(ac)

from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)



df = pd.read_csv(r"D:\vs code fsds\final1.csv")

d2= df.copy()
df= df.iloc[:,[3,4]].values

from sklearn.preprocessing import StandardScaler
classsifier.fit(x_train,y_train)
sc= StandardScaler()

M= sc.fit_transform(df)
print(M)


y_pred1 = pd.DataFrame()
d2 ['y_pred1']= classsifier.predict(M)

d2.to_csv('final1_prediction.csv')


from sklearn.metrics import roc_auc_score,roc_curve
y_pred_prob = classsifier.predict_proba(x_test)[:,1]

auc_score= roc_auc_score(y_test,y_pred_prob)
fpr,tpr,threshold = roc_curve(y_test,y_pred_prob)


plt.figure(figsize=(8,6))
plt.plot(fpr,tpr ,label=f'Logistic Regression(AUC={auc_score:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('Flase Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='Lower right')
plt.grid()
plt.show()

















