import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from utils.data_collector import get_air_quality, process_air_quality_data
import config
from geopy.geocoders import ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Digital Twin Cities",
    page_icon="",
    layout="wide"
)

# --- SESSION STATE ---
if 'zip_code' not in st.session_state:
    st.session_state.zip_code = config.DEFAULT_ZIP
if 'map_center' not in st.session_state:
    st.session_state.map_center = config.MAP_CENTER
if 'last_map_click' not in st.session_state:
    st.session_state.last_map_click = None

# --- HELPER FUNCTIONS ---
def get_zip_from_coords(lat, lon):
    """
    Converts coordinates to a ZIP code using ArcGIS geocoder.
    Returns a tuple: (zip_code, error_message).
    """
    try:
        geolocator = ArcGIS()
        location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
        
        if location and location.raw.get('address', {}).get('Postal'):
            return location.raw['address']['Postal'], None
        elif location and location.raw.get('Postal'):
            return location.raw['Postal'], None
        else:
            return None, "Cannot determine ZIP Code from the selected location."
            
    except (GeocoderTimedOut, GeocoderUnavailable) as e:
        st.error(f"Geocoding service is unavailable. Please check your connection or try again later.")
        return None, f"Network error: {e}"
    except Exception as e:
        st.error(f"An unexpected error occurred during geocoding: {e}")
        return None, f"Unexpected error: {e}"

# --- MAIN APP ---
st.title("Digital Twin Cities for Longevity")
st.markdown("### An interactive dashboard to explore the impact of social determinants of health on aging.")

# Create two columns for the map and the data
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive Map")
    
    m = folium.Map(location=st.session_state.map_center, zoom_start=12)
    
    # Add a marker to show the currently selected location
    folium.Marker(
        location=[st.session_state.map_center[0], st.session_state.map_center[1]],
        popup=f"Data for ZIP: {st.session_state.zip_code}",
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(m)

    map_data = st_folium(m, width=700, height=500, center=st.session_state.map_center)

    # When the map is clicked, update the ZIP code
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        
        # Check if the click location is different from the last one
        if (clicked_lat, clicked_lon) != st.session_state.get('last_map_click'):
            st.session_state.last_map_click = (clicked_lat, clicked_lon)
            
            with st.spinner("Getting ZIP code for new location..."):
                new_zip, error = get_zip_from_coords(clicked_lat, clicked_lon)
            
            if error:
                st.warning(error)
            elif new_zip:
                st.session_state.zip_code = new_zip
                # Update map center to the new location
                st.session_state.map_center = [clicked_lat, clicked_lon]
                st.rerun()

with col2:
    st.subheader(f"Air Quality Data for ZIP: {st.session_state.zip_code}")
    
    with st.spinner(f"Fetching data for {st.session_state.zip_code}..."):
        aq_data = get_air_quality(st.session_state.zip_code)
        
        if aq_data:
            df = process_air_quality_data(aq_data)
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                # Display key metrics
                st.metric(label="Overall AQI", value=df["aqi"].iloc[0], help=df["category"].iloc[0])
                
                # Chart the pollutants
                pollutants_df = df.melt(
                    id_vars=['category'], 
                    value_vars=['aqi'],
                    var_name='Pollutant',
                    value_name='AQI_Value'
                )
                pollutants_df = pollutants_df[pollutants_df['AQI_Value'] > 0] # Filter out non-present pollutants
                
                if not pollutants_df.empty:
                    st.bar_chart(pollutants_df.set_index('Pollutant')['AQI_Value'])
                else:
                    st.info("No primary pollutant data available for this location.")

            else:
                st.warning("No air quality data could be processed for this ZIP code. It might be a rural or invalid ZIP.")
        else:
            st.error(f"Could not retrieve air quality data. Please check the ZIP code or your API key.") 