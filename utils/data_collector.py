import requests
import pandas as pd
from datetime import datetime
import config

def get_air_quality(zip_code):
    """
    Fetch current air quality data from AirNow API for a given zip code.
    
    Args:
        zip_code (str): ZIP code to fetch data for
        
    Returns:
        dict: Air quality data including AQI and pollutant information
    """
    try:
        url = f"{config.AIRNOW_BASE_URL}?format=application/json&zipCode={zip_code}&API_KEY={config.AIRNOW_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            return None
            
        # Process the first observation (most recent)
        observation = data[0]
        
        date_observed = observation['DateObserved']
        hour_observed = observation.get('HourObserved', '00') # Default to '00' if not present
        
        # Combine date and hour into a single string for parsing
        date_time_str = f"{date_observed} {hour_observed}:00:00"
        
        return {
            'timestamp': datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S'),
            'aqi': observation['AQI'],
            'category': observation['Category']['Name'],
            'pollutant': observation['ParameterName'],
            'zip_code': zip_code
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching air quality data: {e}")
        return None

def get_historical_air_quality(zip_code, start_date, end_date):
    """
    Fetch historical air quality data for a given zip code and date range.
    
    Args:
        zip_code (str): ZIP code to fetch data for
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        
    Returns:
        pd.DataFrame: Historical air quality data
    """
    # Note: This is a placeholder. The actual implementation would depend on
    # the specific API endpoints available for historical data
    pass

def process_air_quality_data(data):
    """
    Process raw air quality data into a format suitable for analysis.
    
    Args:
        data (dict): Raw air quality data
        
    Returns:
        pd.DataFrame: Processed air quality data
    """
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame([data])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df 