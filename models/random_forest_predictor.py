from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class RandomForestPredictor:
    def __init__(self, mode='regression', params=None):
        self.mode = mode
        self.model = (RandomForestRegressor(**params) if mode == 'regression'
                      else RandomForestClassifier(**params))
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)