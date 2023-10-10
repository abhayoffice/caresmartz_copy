from sklearn.metrics import r2_score
from pmdarima.arima import auto_arima
import pandas as pd

class SARIMAModel:
    def get_score(self, test, sarima_forecast):
        sarima_forecast_series = pd.Series(sarima_forecast, index=test.index)
        r2_sarima = r2_score(test["Invoice Amount"], sarima_forecast_series)
        return r2_sarima

    def get_model(self, train, test):
        print("------------------------------The Sarima Model ------------------------------")
        sarima_model = auto_arima(train, seasonal=True, m=12, stepwise=True, trace=False)
        return sarima_model

    def get_forecast(self, train ,test , model):
        sarima_forecast = model.predict(n_periods=len(test))
        return sarima_forecast


