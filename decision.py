import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"D:\emp_sal.csv")

x= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
dt_reg= DecisionTreeRegressor(splitter='best',criterion='poisson',max_depth=3)
dt_reg.fit(x,y)
dt_reg_pred=dt_reg.predict([[6.5]])
print(dt_reg_pred)




