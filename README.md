# Digital Twin Cities — Ultra Dashboard

A next-generation, ultra-modern analytics platform for exploring, modeling, and impacting Social Determinants of Health (SDOH) across the US. Featuring a stunning glassmorphic UI, animated charts, interactive maps, and advanced machine learning insights.

---

## 🚀 Overview

Digital Twin Cities is a visually immersive dashboard that empowers users to:
- Explore SDOH data at the census tract level
- Visualize and interact with geospatial health determinants
- Run advanced ML models and see feature importance
- Simulate policy changes and assess health impact
- Experience a beautiful, dark, glassy, and animated interface

---

## ✨ Key Features

- **Custom Top Navigation** — Smooth, beautiful tab switching (Overview, Map, Explorer, ML Insights, Impact)
- **Glassmorphism & Gradients** — Modern, premium look throughout
- **Full-Width Interactive Map** — Clickable, color-coded tracts with floating info panels
- **Animated Charts** — Histograms, box plots, and correlation matrices with stunning color palettes
- **ML Insights** — Model comparison, feature importance, and prediction accuracy
- **Policy Impact Calculator** — Instantly see the effect of SDOH changes
- **Responsive & Fast** — Works on all screen sizes, optimized for performance

---

## 🛠️ Setup

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

3. **Set up environment variables:**
Create a `.env` file in the root directory with your API keys:
```
AIRNOW_API_KEY=your_airnow_api_key
```

4. **Run the ultra dashboard:**
```bash
streamlit run ultra_dashboard.py
```

---

## 📁 Project Structure

```
Digital-Twin-Cities-3/
├── ultra_dashboard.py        # Ultra-modern glassy dashboard (recommended)
├── app.py                   # Original Streamlit app
├── models/                  # ML model implementations
├── utils/                   # Utility functions
├── config.py                # Configuration settings
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📊 Data Sources
- EPA AirNow API: Air quality data
- U.S. Census Bureau ACS: Socioeconomic indicators
- CDC PLACES: Health outcome data
- USDA Food Access Research Atlas: Food access information
- OpenStreetMap: Geospatial data

---

## 🤝 Contributing
Contributions are welcome! Please submit a Pull Request or open an Issue.

---

## 📄 License
MIT License — see the LICENSE file for details.