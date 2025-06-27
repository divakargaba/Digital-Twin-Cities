from sklearn.metrics import mean_squared_error, r2_score

def print_regression_metrics(y_true, y_pred, model_name="Model"):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name} Evaluation:\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}")