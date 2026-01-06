import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"D:\emp_sal.csv")
x= dataset.iloc[:,1:2]
y= dataset.iloc[:,2]

from sklearn.svm import SVR
regressor= SVR(kernel='rbf',degree=3)
regressor.fit(x,y)
y_pred_svr= regressor.predict([[6.5]])
print(y_pred_svr)


from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(weights='distance')
knn_reg.fit(x,y)
y_pred_knn= knn_reg.predict([[6.5]])
print(y_pred_knn)





