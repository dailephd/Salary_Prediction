import os
import requests
import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape


def stage1(data):
    X = data["rating"]
    y = data["salary"]
    # Split predictor and target into training and test parts. Use test_size=0.3 and random_state=100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    # Fit the linear regression model with the following formula on the training data: salary∼rating
    linReg = LinearRegression().fit(X_train, y_train)
    # Predict a salary with the fitted model on test data and calculate the MAPE;
    y_pred = linReg.predict(X_test)
    # Calculate MAPE
    MAPE = mape(y_test, y_pred).round(5)
    print(linReg.intercept_.round(5), linReg.coef_[0].round(5), MAPE.round(5))


def stage2(data):
    X = data["rating"]
    y = data["salary"]
    X2 = np.power(X, 2)
    X3 = np.power(X, 3)
    X4 = np.power(X, 4)
    Z = [X2, X3, X4]
    mapelist = []
    for x in Z:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
        linReg = LinearRegression().fit(X_train, y_train)
        y_pred = linReg.predict(X_test)
        mapelist.append(mape(y_test, y_pred).round(5))
    print(min(mapelist))


def stage3(data):
    X = data.loc[:, data.columns != 'salary']
    y = data['salary']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)
    linReg = LinearRegression().fit(X_train, y_train)
    y_pred = linReg.predict(X_test)
    coef = linReg.coef_.round(5).item()
    print(coef)


def stage4(data):

    # Find the variables where the correlation coefficient is greater than 0.2
    # print(self.data.corr())
    X = data.drop(["salary"], axis=1)
    y = data["salary"]
    cols1 = ["rating", "age", "experience"]
    # Get all combinations length 2 of cols
    cols2 = [list(tup) for tup in itertools.combinations(cols1, 2)]
    # Combine cols1 and cols 2 into 1 list of possible combinations of variables
    cols = cols1 + cols2
    mapelist = []
    for i in cols:
        X1 = X.drop(i, axis=1)
        # Split predictor and target into training and test parts. Use test_size=0.3 and random_state=100
        X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=100)
        # Fit the linear regression model with the following formula on the training data: salary∼rating
        linReg = LinearRegression().fit(X_train, y_train)
        # Predict a salary with the fitted model on test data and calculate the MAPE;
        y_pred = linReg.predict(X_test)
        mapelist.append(mape(y_test, y_pred).round(5))
   #print(mapelist)
    # Get index of min mape in mapelist
    minidx = np.argmin(mapelist)
    # Return columns which can be eliminated for best prediction
    return cols[minidx]


def stage5(data):
    X = data.drop(["salary"], axis=1)
    y = data["salary"]
    # Get columns with best prediction from previous stage
    dropcols = stage4(data)
    X = X.drop(dropcols, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    linReg = LinearRegression().fit(X_train, y_train)
    y_pred = linReg.predict(X_test)
    MAPE= mape(y_test,  y_pred).round(5)
    # Copy y_pred
    new_ypred = np.copy(y_pred)
    new_ypred2 = np.copy(y_pred)
    # Get indexes of values < 0
    negidx = np.where(y_pred < 0)
    for i in negidx:
        # Option 1: replace the negative values with 0
        new_ypred[negidx] = 0
        # Option 2 : replace the negative values with the median of the training part of y
        new_ypred2[negidx] = np.median(y_train)
    MAPE1 = mape(y_test,  new_ypred).round(5)
    MAPE2 = mape(y_test,  new_ypred2).round(5)
    print(min(MAPE, MAPE1, MAPE2))


def main():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # download data if it is unavailable
    if 'data.csv' not in os.listdir('../Data'):
        url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/data.csv', 'wb').write(r.content)

    data = pd.read_csv('../Data/data.csv')
    # stage1(data)
    # stage2(data)
    # stage3(data)
    # stage4(data)
    stage5(data)

if __name__ == '__main__':
    main()