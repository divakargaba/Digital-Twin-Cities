from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

class ANNPredictor:
    def __init__(self, params):
        self.model = MLPRegressor(**params)
        self.scaler = StandardScaler()

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)