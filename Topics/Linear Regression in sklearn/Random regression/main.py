#  write your code here 
import pandas as pd
from sklearn.linear_model import LinearRegression
path = "C:/Users/daile/PycharmProjects/Salary Prediction/Topics/Linear Regression in sklearn/Random regression/data/dataset/input.txt"
df = pd.read_csv(path)
X_train = df.iloc[:-70,:4]
X_test = df.iloc[-70:, :4]
y_train = df.target[:-70]
y_test = df.target[-70:]
linReg = LinearRegression().fit(X_train, y_train)
print(linReg.intercept_.round(3))