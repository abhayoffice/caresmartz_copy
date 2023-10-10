from statsmodels.tsa.api import AutoReg
from sklearn.metrics import r2_score
import pandas as pd

class AutoRegModel:
    def get_score(self, test, autoreg_forecast):
        autoreg_forecast_series = pd.Series(autoreg_forecast, index=test.index)
        r2_autoreg = r2_score(test["Invoice Amount"], autoreg_forecast_series)
        return r2_autoreg

    def get_model(self, train, test):
        # AutoReg model
        print("------------------------------The AutoReg Model ------------------------------")
        autoreg_model = AutoReg(train, lags=1)
        autoreg_model_fit = autoreg_model.fit()
        return autoreg_model_fit

    def get_forecast(self, train, test, model):
        autoreg_forecast = model.predict(start=len(train), end=len(train) + len(test) - 1)
        return autoreg_forecast

