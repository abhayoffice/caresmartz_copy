import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class LinearRegressionModel:
    def __init__(self):
        self.X_test_data , self.y_test_data = None , None

    def get_score(self, test , pred):
        r2 = r2_score(self.y_test_data, pred)
        return r2

    def get_model(self, train , test):
        print("------------------------------The LinearRegression Model ------------------------------")

        train ['data_1month'] = train['Invoice Amount'].shift(+1)
        train ['data_2month'] = train['Invoice Amount'].shift(+2)
        train ['data_3month'] = train['Invoice Amount'].shift(+3)

        test['data_1month'] = test['Invoice Amount'].shift(+1)
        test['data_2month'] = test['Invoice Amount'].shift(+2)
        test['data_3month'] = test['Invoice Amount'].shift(+3)

        train = train.dropna()
        test = test.dropna()

        y_train = np.array(train["Invoice Amount"])
        y_test = np.array(test["Invoice Amount"])

        x1_train = np.array(train['data_1month'])
        x2_train = np.array(train['data_2month'])
        x3_train = np.array(train['data_3month'])

        x1_test = np.array(test['data_1month'])
        x2_test = np.array(test['data_2month'])
        x3_test = np.array(test['data_3month'])

        x1_train, x2_train, x3_train, y_train = x1_train.reshape(-1,1) , x2_train.reshape(-1,1), x3_train.reshape(-1,1), y_train.reshape(-1,1)
        x1_test, x2_test, x3_test, y_test = x1_test.reshape(-1,1) , x2_test.reshape(-1,1), x3_test.reshape(-1,1), y_test.reshape(-1,1)

        final_x_train = np.concatenate((x1_train,x2_train,x3_train), axis=1)
        final_x_test = np.concatenate((x1_test,x2_test,x3_test), axis=1)

        self.x_test_data = final_x_test
        self.y_test_data = y_test
        reg_model = LinearRegression()
        model = reg_model.fit(final_x_train, y_train)

        return model

    def get_forecast(self, test ,train, model):
        pred = model.predict(self.x_test_data)
        return pred

