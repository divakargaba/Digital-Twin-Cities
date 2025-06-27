# Digital Twin Cities — Analytics & Simulation Platform

Digital Twin Cities is an advanced analytics platform for modeling, analyzing, and simulating the impact of Social Determinants of Health (SDOH) across the United States at the census tract level. The project leverages large-scale, multi-source datasets and state-of-the-art machine learning to provide actionable insights for researchers, policymakers, and public health professionals.

---

## Overview

This platform enables users to:
- Explore and visualize SDOH and health outcome data at a granular, neighborhood level
- Analyze the relationships between environmental, social, and economic factors and health outcomes
- Run and compare advanced machine learning models for health risk prediction
- Simulate policy interventions and instantly assess projected impacts on community health
- Integrate and process large, heterogeneous datasets from authoritative sources

---

## Key Features

- **Comprehensive Data Integration:** Ingests and processes data from the US Census ACS, CDC PLACES, EPA AirNow, USDA Food Access, and more
- **Robust ML Pipeline:** Feature engineering, missing value handling, and encoding for SDOH data
- **Multiple Predictive Models:** Includes Artificial Neural Networks (ANN), Graph Neural Networks (GNN), Random Forests, XGBoost, and custom health risk predictors
- **Scenario Simulation:** Users can adjust SDOH variables and simulate the impact of policy changes on health risk metrics
- **Interactive Analytics:** Visualizations, geospatial mapping, and model evaluation tools
- **Scalable Architecture:** Designed for extensibility and integration of new data sources and models

---

## Setup

1. **Clone the repository:**
```bash
git clone [repository-url]
cd Digital-Twin-Cities-3
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install streamlit-option-menu
```

3. **Set up environment variables (if needed):**
Create a `.env` file in the root directory with your API keys:
```
AIRNOW_API_KEY=your_airnow_api_key
```

4. **Run the dashboard:**
```bash
streamlit run ultra_dashboard.py
```

---

## Project Structure

```
Digital-Twin-Cities-3/
├── ultra_dashboard.py        # Main dashboard app
├── app.py                   # Original Streamlit app
├── models/                  # ML model implementations
├── utils/                   # Utility functions
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## Data Sources
- EPA AirNow API: Air quality data
- U.S. Census Bureau ACS: Socioeconomic indicators
- CDC PLACES: Health outcome data
- USDA Food Access Research Atlas: Food access information
- OpenStreetMap: Geospatial data

---

## Team
- **Divakar Gaba** — Lead AI Engineer
- **AbdulRahman Negmeldin** — Data Scientist
- **Omar Negmeldin** — Researcher
- **Wadi Alam** — Researcher

---

## License
MIT License — see the LICENSE file for details.