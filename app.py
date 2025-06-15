import streamlit as st
import folium
from streamlit_folium import folium_static
import pandas as pd
from utils.data_collector import get_air_quality, process_air_quality_data
import config

# Set page config
st.set_page_config(
    page_title="Digital Twin Cities",
    page_icon="🏙️",
    layout="wide"
)

# Title and description
st.title("🏙️ Digital Twin Cities for Longevity")
st.markdown("""
This application simulates how urban social determinants of health (SDOH) affect aging outcomes 
and chronic disease progression. Explore how changes in air quality, walkability, and other factors 
impact population health.
""")

# Sidebar
st.sidebar.header("Settings")
zip_code = st.sidebar.text_input("ZIP Code", value=config.DEFAULT_ZIP)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("City Map")
    # Create a map centered on the default location
    m = folium.Map(location=config.MAP_CENTER, zoom_start=config.MAP_ZOOM)
    folium_static(m)

with col2:
    st.subheader("Air Quality Data")
    if zip_code:
        air_quality_data = get_air_quality(zip_code)
        if air_quality_data:
            df = process_air_quality_data(air_quality_data)
            st.write("Current Air Quality Index (AQI):", df['aqi'].iloc[0])
            st.write("Category:", df['category'].iloc[0])
            st.write("Primary Pollutant:", df['pollutant'].iloc[0])
        else:
            st.error("Could not fetch air quality data. Please check the ZIP code and try again.")

# Simulation Controls
st.header("Simulation Controls")
col3, col4, col5 = st.columns(3)

with col3:
    st.subheader("Air Quality")
    air_quality_change = st.slider(
        "Change in Air Quality (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5
    )

with col4:
    st.subheader("Walkability")
    walkability_change = st.slider(
        "Change in Walkability Score",
        min_value=-20,
        max_value=20,
        value=0,
        step=1
    )

with col5:
    st.subheader("Green Space")
    green_space_change = st.slider(
        "Change in Green Space (%)",
        min_value=-30,
        max_value=30,
        value=0,
        step=5
    )

# Results Section
st.header("Simulation Results")
if st.button("Run Simulation"):
    # Placeholder for simulation results
    st.info("Simulation running... This is a placeholder for the actual simulation results.")
    
    # Example results visualization
    results_data = {
        'Metric': ['Life Expectancy', 'Chronic Disease Risk', 'Healthcare Costs'],
        'Current': [75, 25, 100],
        'Projected': [77, 22, 95]
    }
    results_df = pd.DataFrame(results_data)
    st.dataframe(results_df)
    
    # Add a simple line chart
    st.line_chart(results_df.set_index('Metric')[['Current', 'Projected']]) 