import xgboost as xgb
from sklearn.preprocessing import StandardScaler

class XGBoostPredictor:
    def __init__(self, params):
        self.model = xgb.XGBRegressor(**params)
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)