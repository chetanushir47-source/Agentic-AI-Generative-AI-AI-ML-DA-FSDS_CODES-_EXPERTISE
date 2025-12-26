import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"D:\emp_sal.csv")

x= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.ensemble import RandomForestRegressor
r_for= RandomForestRegressor(random_state=43,n_estimators=20)
r_for.fit(x,y)
r_pred= r_for.predict([[6.5]])
print(r_pred)

