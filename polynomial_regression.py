import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\emp_sal.csv")

x= dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

plt.scatter(x,y, color='red')
plt.plot(x, lin_reg.predict(x),color = 'blue')
plt.title('linear Regression graph')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)

# we will be usimg polynomial model because linear model is not handling non linear data

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=5)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y ,color='red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

lin_model_pred= lin_reg.predict([[6.5]])
lin_model_pred 

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

print(poly_model_pred)