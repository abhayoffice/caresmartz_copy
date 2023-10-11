from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score
import pandas as pd
import itertools


class SARIMAModel:

    def get_score(self, test, model_forecast):
        r2_sarima = r2_score(test, model_forecast)
        return r2_sarima

    def get_model(self, train, test):
        print("------------------------------The Seasonal ARIMA Model ------------------------------")

        # Define the range of p, d, q values
        p = range(0, 2)
        q = range(0, 2)
        d = range(0, 2)

        # Create all possible combinations of p, d, q
        pdq = list(itertools.product(p, d, q))
        model_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        # Define an empty data frame to store the parameter values along with the model AIC
        SARIMA_AIC = pd.DataFrame(columns=['param', 'seasonal', 'AIC'])

        for param in pdq:
            for param_seasonal in model_pdq:
                try:
                    SARIMA_model = SARIMAX(train, order=param, seasonal_order=param_seasonal)
                    results_SARIMA = SARIMA_model.fit(disp=False)
                except:
                    continue

                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results_SARIMA.aic))
                SARIMA_AIC = SARIMA_AIC._append({'param': param, 'seasonal': param_seasonal, 'AIC': results_SARIMA.aic},
                                               ignore_index=True)

        # Sort the SARIMA_AIC DataFrame by AIC and get the parameters with the lowest AIC
        best_parameters = SARIMA_AIC.sort_values(by=['AIC']).iloc[0]
        print("Best Parameters (p, d, q, P, D, Q, S):", best_parameters['param'], best_parameters['seasonal'])
        print("Lowest AIC:", best_parameters['AIC'])

        # Train and return the SARIMA model with the best parameters
        sarima_model = SARIMAX(train, order=best_parameters['param'], seasonal_order=best_parameters['seasonal']).fit(
            disp=0)
        return sarima_model

    def get_forecast(self, train, test, model):
        sarima_forecast = model.forecast(steps=len(test))
        return sarima_forecast
