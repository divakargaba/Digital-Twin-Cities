import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
AIRNOW_API_KEY = os.getenv('AIRNOW_API_KEY')

# API Endpoints
AIRNOW_BASE_URL = "https://www.airnowapi.org/aq/observation/zipCode/current/"

# Data Settings
DEFAULT_CITY = "New York"
DEFAULT_STATE = "NY"
DEFAULT_ZIP = "10001"

# Model Settings
MODEL_PARAMS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
}

# Visualization Settings
MAP_CENTER = [40.7128, -74.0060]  # Default to NYC
MAP_ZOOM = 12 