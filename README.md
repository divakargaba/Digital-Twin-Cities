# Digital Twin Cities for Longevity

A machine learning-powered simulation platform that models how urban social determinants of health (SDOH) affect aging outcomes and chronic disease progression.

## Overview

This project creates digital twin simulations of urban environments to model the impact of changes in social determinants of health (SDOH) on aging populations. By integrating environmental and epidemiological data, it predicts how variations in factors like air quality, walkability, and food access influence longevity and chronic disease progression.

## Features

- Real-time air quality monitoring and historical data analysis
- Interactive visualization of SDOH factors and their health impacts
- Machine learning models for predicting health outcomes
- Simulation of urban policy changes and their effects
- Geospatial analysis of neighborhood-level health determinants

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd Digital-Twin-Cities
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with your API keys:
```
AIRNOW_API_KEY=your_airnow_api_key
```

4. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
Digital-Twin-Cities/
├── app.py                 # Main Streamlit application
├── data/                  # Data storage directory
├── models/               # ML model implementations
├── utils/                # Utility functions
├── config.py             # Configuration settings
└── requirements.txt      # Project dependencies
```

## Data Sources

- EPA AirNow API: Air quality data
- U.S. Census Bureau ACS: Socioeconomic indicators
- CDC PLACES: Health outcome data
- USDA Food Access Research Atlas: Food access information
- OpenStreetMap: Geospatial data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.