from statsmodels.tsa.api import AutoReg
from sklearn.metrics import r2_score
import pandas as pd

class AutoRegModel:
    def __init__(self, config):
        self.config = config

    def get_score(self, test, autoreg_forecast):
        autoreg_forecast_series = pd.Series(autoreg_forecast, index=test.index)
        r2_autoreg = r2_score(test[self.config["target_column"]], autoreg_forecast_series)
        return r2_autoreg

    def get_model(self, train, test):
        print("------------------------------The AutoReg Model ------------------------------")
        autoreg_model = AutoReg(train[self.config["target_column"]], lags=self.config["lags"])
        autoreg_model_fit = autoreg_model.fit()
        return autoreg_model_fit

    def get_forecast(self, train, test, model):
        autoreg_forecast = model.predict(start=len(train), end=len(train) + len(test) - 1)
        return autoreg_forecast
