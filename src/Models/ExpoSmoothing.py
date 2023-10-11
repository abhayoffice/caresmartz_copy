from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import r2_score

class ESModel:

    def get_score(self, test, model_forecast):
        #To check and fill the missing values (If any)
        test.fillna(test.mean(), inplace=True)
        r2_expo_smoothing = r2_score(test['Invoice Amount'], model_forecast)
        return r2_expo_smoothing

    def get_model(self, train, test):
        print("------------------------------The Exponential Smoothing Model ------------------------------")
        # Simple Exponential Smoothing (SES) model
        ses_model = SimpleExpSmoothing(train['Invoice Amount']).fit()
        return ses_model

    def get_forecast(self, train, test, model):
        expo_smoothing_forecast = model.forecast(steps=len(test))
        return expo_smoothing_forecast