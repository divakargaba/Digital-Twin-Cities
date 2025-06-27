from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from utils.evaluation import print_regression_metrics

class XGBoostPredictor:
    def __init__(self, params):
        self.model = xgb.XGBRegressor(**params)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, X_test=None, y_test=None):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

        if X_test is not None and y_test is not None:
            preds = self.predict(X_test)
            print_regression_metrics(y_test, preds, model_name="XGBoost")

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
