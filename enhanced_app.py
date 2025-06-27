import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_collector import get_air_quality, process_air_quality_data
from utils.data_processor import SDOHDataProcessor
from ml_pipeline import SDOHMLPipeline
import config
from geopy.geocoders import ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Digital Twin Cities - SDOH Analysis",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #1f2937, #374151);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: #f8fafc;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Session State Management
if 'zip_code' not in st.session_state:
    st.session_state.zip_code = config.DEFAULT_ZIP
if 'map_center' not in st.session_state:
    st.session_state.map_center = config.MAP_CENTER
if 'last_map_click' not in st.session_state:
    st.session_state.last_map_click = None
if 'sdoh_data_loaded' not in st.session_state:
    st.session_state.sdoh_data_loaded = False
if 'ml_pipeline' not in st.session_state:
    st.session_state.ml_pipeline = None
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None

# Helper Functions
@st.cache_data
def load_sdoh_data():
    """Load and cache SDOH data."""
    try:
        processor = SDOHDataProcessor("Non-Medical_Factor_Measures_for_Census_Tract__ACS_2017-2021_20250626.csv")
        processor.load_data(sample_size=50000)
        processor.explore_data()
        processor.pivot_data()
        processor.handle_missing_values(strategy='drop')
        processor.encode_categorical_features()
        processor.create_target_variable(target_type='health_index')
        return processor
    except Exception as e:
        st.error(f"Error loading SDOH data: {e}")
        return None

def get_zip_from_coords(lat, lon):
    """Convert coordinates to ZIP code."""
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
        return None, f"Network error: {e}"
    except Exception as e:
        return None, f"Unexpected error: {e}"

def create_interactive_map(data_processor, selected_state=None):
    """Create an interactive map with SDOH data and click functionality."""
    if data_processor is None or data_processor.processed_df is None:
        return None
        
    # Filter data by state if specified
    map_data = data_processor.processed_df.copy()
    if selected_state:
        map_data = map_data[map_data['StateAbbr'] == selected_state]
        
    # Sample data for better performance
    if len(map_data) > 2000:
        map_data = map_data.sample(n=2000, random_state=42)
        
    # Create map
    center_lat = map_data['latitude'].mean()
    center_lon = map_data['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='cartodbpositron')
    
    # Add markers for each census tract
    for _, row in map_data.iterrows():
        # Color based on health index
        health_index = row.get('health_index', 0)
        if pd.isna(health_index):
            color = '#6b7280'  # Gray
        elif health_index > 50:
            color = '#dc2626'  # Red
        elif health_index > 25:
            color = '#f59e0b'  # Orange
        else:
            color = '#16a34a'  # Green
            
        # Create detailed popup content
        popup_content = f"""
        <div style="width: 300px; font-family: Arial, sans-serif;">
            <h4 style="margin: 0 0 10px 0; color: #1f2937;">Census Tract: {row['LocationID']}</h4>
            <p style="margin: 5px 0; color: #374151;"><strong>Location:</strong> {row['CountyName']}, {row['StateAbbr']}</p>
            <p style="margin: 5px 0; color: #374151;"><strong>Population:</strong> {row['TotalPopulation']:,}</p>
            <hr style="margin: 10px 0; border: 1px solid #e5e7eb;">
            <h5 style="margin: 10px 0 5px 0; color: #1f2937;">Health Index: {health_index:.1f}</h5>
            <div style="background: #f3f4f6; padding: 10px; border-radius: 5px;">
                <p style="margin: 3px 0; font-size: 12px;"><strong>SDOH Factors:</strong></p>
                <p style="margin: 2px 0; font-size: 11px;">• Crowding: {row.get('crowding_pct', 'N/A'):.1f}%</p>
                <p style="margin: 2px 0; font-size: 11px;">• Housing Cost Burden: {row.get('housing_cost_burden_pct', 'N/A'):.1f}%</p>
                <p style="margin: 2px 0; font-size: 11px;">• No Broadband: {row.get('no_broadband_pct', 'N/A'):.1f}%</p>
                <p style="margin: 2px 0; font-size: 11px;">• Unemployment: {row.get('unemployment_pct', 'N/A'):.1f}%</p>
                <p style="margin: 2px 0; font-size: 11px;">• Poverty (150%): {row.get('poverty_150_pct', 'N/A'):.1f}%</p>
            </div>
        </div>
        """
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            popup=folium.Popup(popup_content, max_width=350),
            color=color,
            fill=True,
            fillOpacity=0.7,
            weight=1
        ).add_to(m)
        
    return m

def create_location_analysis(data_processor, lat, lon):
    """Analyze data for a specific location."""
    if data_processor is None or data_processor.processed_df is None:
        return None
        
    # Find nearest census tract
    data = data_processor.processed_df.copy()
    data['distance'] = np.sqrt((data['latitude'] - lat)**2 + (data['longitude'] - lon)**2)
    nearest = data.loc[data['distance'].idxmin()]
    
    return nearest

def create_animated_charts(data_processor, selected_state=None):
    """Create animated and interactive charts for SDOH analysis."""
    if data_processor is None or data_processor.processed_df is None:
        return None, None, None
        
    # Filter data
    chart_data = data_processor.processed_df.copy()
    if selected_state:
        chart_data = chart_data[chart_data['StateAbbr'] == selected_state]
        
    # 1. Animated Health Index Distribution
    fig1 = px.histogram(
        chart_data, 
        x='health_index', 
        nbins=30,
        title=f'Distribution of Health Index {"by " + selected_state if selected_state else "Nationwide"}',
        labels={'health_index': 'Health Index', 'count': 'Number of Census Tracts'},
        animation_frame='StateAbbr' if selected_state is None else None,
        color_discrete_sequence=['#3b82f6']
    )
    fig1.update_layout(
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12)
    )
    
    # 2. Interactive Correlation Heatmap
    sdoh_cols = ['crowding_pct', 'housing_cost_burden_pct', 'no_broadband_pct', 
                 'no_hs_diploma_pct', 'poverty_150_pct', 'unemployment_pct', 
                 'minority_pct', 'single_parent_pct', 'seniors_pct']
    
    correlation_data = chart_data[sdoh_cols + ['health_index']].corr()
    
    fig2 = px.imshow(
        correlation_data,
        title='SDOH Factors Correlation Matrix',
        color_continuous_scale='RdBu',
        aspect='auto',
        text_auto=True
    )
    fig2.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=10)
    )
    
    # 3. Interactive Box Plot
    if selected_state is None:
        state_avg = chart_data.groupby('StateAbbr')[sdoh_cols].mean().reset_index()
        state_avg_melted = state_avg.melt(
            id_vars=['StateAbbr'], 
            value_vars=sdoh_cols,
            var_name='SDOH_Factor', 
            value_name='Percentage'
        )
        
        fig3 = px.box(
            state_avg_melted,
            x='SDOH_Factor',
            y='Percentage',
            title='SDOH Factors Distribution Across States',
            labels={'SDOH_Factor': 'Social Determinant of Health Factor', 'Percentage': 'Percentage (%)'},
            color='SDOH_Factor',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    else:
        county_avg = chart_data.groupby('CountyName')[sdoh_cols].mean().reset_index()
        county_avg_melted = county_avg.melt(
            id_vars=['CountyName'], 
            value_vars=sdoh_cols,
            var_name='SDOH_Factor', 
            value_name='Percentage'
        )
        
        fig3 = px.box(
            county_avg_melted,
            x='SDOH_Factor',
            y='Percentage',
            title=f'SDOH Factors Distribution Across Counties in {selected_state}',
            labels={'SDOH_Factor': 'Social Determinant of Health Factor', 'Percentage': 'Percentage (%)'},
            color='SDOH_Factor',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    
    fig3.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=10)
    )
    fig3.update_xaxes(tickangle=45)
        
    return fig1, fig2, fig3

def create_ml_insights_dashboard(ml_pipeline):
    """Create comprehensive ML insights dashboard."""
    if ml_pipeline is None:
        return None, None, None
        
    # Model comparison
    comparison_df = ml_pipeline.compare_models()
    
    # Feature importance
    importance_df = ml_pipeline.feature_importance_analysis('random_forest')
    
    # Create model performance chart
    if comparison_df is not None:
        fig_performance = px.bar(
            comparison_df,
            x='Model',
            y='R²',
            title='Model Performance Comparison (R² Score)',
            color='R²',
            color_continuous_scale='Viridis',
            text='R²'
        )
        fig_performance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig_performance.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
    else:
        fig_performance = None
    
    # Create feature importance chart
    if importance_df is not None:
        fig_importance = px.bar(
            importance_df.head(10),
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Most Important Features',
            color='importance',
            color_continuous_scale='Blues'
        )
        fig_importance.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
    else:
        fig_importance = None
    
    # Create predictions vs actual chart
    if hasattr(ml_pipeline, 'results') and 'random_forest' in ml_pipeline.results:
        results = ml_pipeline.results['random_forest']
        fig_predictions = px.scatter(
            x=results['y_test'],
            y=results['y_pred'],
            title='Predictions vs Actual Values',
            labels={'x': 'Actual Health Index', 'y': 'Predicted Health Index'},
            trendline='ols'
        )
        fig_predictions.add_shape(
            type='line',
            x0=results['y_test'].min(),
            y0=results['y_test'].min(),
            x1=results['y_test'].max(),
            y1=results['y_test'].max(),
            line=dict(dash='dash', color='red')
        )
        fig_predictions.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
    else:
        fig_predictions = None
        
    return fig_performance, fig_importance, fig_predictions

# Main Application
st.markdown('<h1 class="main-header">Digital Twin Cities</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Social Determinants of Health Analysis Platform</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Analysis Controls")
    
    # Data loading
    if not st.session_state.sdoh_data_loaded:
        if st.button("Load SDOH Data", key="load_data"):
            with st.spinner("Loading SDOH data..."):
                data_processor = load_sdoh_data()
                if data_processor:
                    st.session_state.data_processor = data_processor
                    st.session_state.sdoh_data_loaded = True
                    st.success("SDOH data loaded successfully!")
                    st.rerun()
    else:
        st.success("SDOH data loaded")
        
    # State selection
    if st.session_state.sdoh_data_loaded:
        states = ['All States'] + sorted(st.session_state.data_processor.processed_df['StateAbbr'].unique().tolist())
        selected_state = st.selectbox("Select State", states)
        if selected_state == 'All States':
            selected_state = None
            
    # ML Pipeline
    st.header("Machine Learning")
    if st.button("Run ML Pipeline", key="run_ml"):
        if st.session_state.sdoh_data_loaded:
            with st.spinner("Running ML pipeline..."):
                try:
                    pipeline = SDOHMLPipeline("Non-Medical_Factor_Measures_for_Census_Tract__ACS_2017-2021_20250626.csv")
                    pipeline.data_processor = st.session_state.data_processor
                    pipeline.prepare_models()
                    pipeline.train_all_models(target_col='health_index')
                    st.session_state.ml_pipeline = pipeline
                    st.success("ML pipeline completed!")
                except Exception as e:
                    st.error(f"Error in ML pipeline: {e}")

# Main content
if st.session_state.sdoh_data_loaded:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Interactive Map", "SDOH Analysis", "ML Insights", "Health Impact"])
    
    with tab1:
        st.header("Interactive Geographic Analysis")
        
        # Create two columns
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create map
            map_obj = create_interactive_map(st.session_state.data_processor, selected_state)
            if map_obj:
                map_data = st_folium(map_obj, width=800, height=600)
                
                # Handle map clicks
                if map_data and map_data.get("last_clicked"):
                    clicked_lat = map_data["last_clicked"]["lat"]
                    clicked_lon = map_data["last_clicked"]["lng"]
                    
                    if (clicked_lat, clicked_lon) != st.session_state.get('last_map_click'):
                        st.session_state.last_map_click = (clicked_lat, clicked_lon)
                        
                        with st.spinner("Analyzing location..."):
                            location_data = create_location_analysis(st.session_state.data_processor, clicked_lat, clicked_lon)
                            if location_data is not None:
                                st.session_state.selected_location = location_data
                                st.rerun()
            else:
                st.warning("No data available for mapping")
        
        with col2:
            st.subheader("Map Legend")
            st.markdown("""
            **Health Index Categories:**
            - 🟢 **Green**: Good health index (0-25)
            - 🟠 **Orange**: Moderate health index (25-50)
            - 🔴 **Red**: Poor health index (>50)
            - ⚫ **Gray**: Missing data
            """)
            
            if st.session_state.selected_location is not None:
                location = st.session_state.selected_location
                st.subheader("Selected Location Analysis")
                
                # Health index gauge
                health_index = location['health_index']
                if health_index <= 25:
                    color = "#16a34a"
                    status = "Good"
                elif health_index <= 50:
                    color = "#f59e0b"
                    status = "Moderate"
                else:
                    color = "#dc2626"
                    status = "Poor"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Health Index</h3>
                    <h2 style="color: {color};">{health_index:.1f}</h2>
                    <p>{status} Health Status</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Location details
                st.markdown(f"""
                **Location Details:**
                - Census Tract: {location['LocationID']}
                - County: {location['CountyName']}, {location['StateAbbr']}
                - Population: {location['TotalPopulation']:,}
                """)
                
                # Key SDOH factors
                st.subheader("Key SDOH Factors")
                factors = [
                    ('Crowding', location.get('crowding_pct', 0)),
                    ('Housing Cost Burden', location.get('housing_cost_burden_pct', 0)),
                    ('No Broadband', location.get('no_broadband_pct', 0)),
                    ('Unemployment', location.get('unemployment_pct', 0)),
                    ('Poverty (150%)', location.get('poverty_150_pct', 0))
                ]
                
                for factor, value in factors:
                    st.metric(factor, f"{value:.1f}%")
    
    with tab2:
        st.header("Social Determinants of Health Analysis")
        
        # Summary statistics
        if selected_state:
            state_data = st.session_state.data_processor.processed_df[
                st.session_state.data_processor.processed_df['StateAbbr'] == selected_state
            ]
        else:
            state_data = st.session_state.data_processor.processed_df
            
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Census Tracts</h3>
                <h2>{len(state_data):,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            avg_health = state_data['health_index'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Average Health Index</h3>
                <h2>{avg_health:.1f}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            total_pop = state_data['TotalPopulation'].sum()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Population</h3>
                <h2>{total_pop:,}</h2>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            high_risk = len(state_data[state_data['health_index'] > 50])
            st.markdown(f"""
            <div class="metric-card">
                <h3>High Risk Areas</h3>
                <h2>{high_risk:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Charts
        fig1, fig2, fig3 = create_animated_charts(st.session_state.data_processor, selected_state)
        
        if fig1 and fig2 and fig3:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig3, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.header("Machine Learning Insights")
        
        if st.session_state.ml_pipeline:
            fig_perf, fig_imp, fig_pred = create_ml_insights_dashboard(st.session_state.ml_pipeline)
            
            if fig_perf:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_perf, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if fig_imp:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_imp, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if fig_pred:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_pred, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            # ML insights text
            st.markdown("""
            <div class="info-box">
                <h4>Key Insights:</h4>
                <ul>
                    <li><strong>Model Performance:</strong> Compare different ML algorithms to understand which performs best for health index prediction</li>
                    <li><strong>Feature Importance:</strong> Identify the most critical social determinants affecting health outcomes</li>
                    <li><strong>Prediction Accuracy:</strong> Visualize how well the models predict actual health indices</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Run the ML pipeline from the sidebar to see insights")
    
    with tab4:
        st.header("Health Impact Assessment")
        
        # Interactive health impact calculator
        st.subheader("Policy Impact Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Conditions**")
            crowding = st.slider("Crowding (%)", 0.0, 20.0, 5.0, key="crowding_current")
            housing_cost = st.slider("Housing Cost Burden (%)", 0.0, 50.0, 25.0, key="housing_current")
            broadband = st.slider("No Broadband (%)", 0.0, 30.0, 10.0, key="broadband_current")
            
        with col2:
            st.markdown("**Proposed Changes**")
            crowding_change = st.slider("Crowding Change (%)", -5.0, 5.0, 0.0, key="crowding_change")
            housing_change = st.slider("Housing Cost Change (%)", -10.0, 10.0, 0.0, key="housing_change")
            broadband_change = st.slider("Broadband Change (%)", -10.0, 10.0, 0.0, key="broadband_change")
            
        # Calculate health impact
        if st.button("Calculate Health Impact", key="calculate_impact"):
            # Simple health impact calculation
            current_health = crowding * 0.3 + housing_cost * 0.4 + broadband * 0.3
            new_health = (crowding + crowding_change) * 0.3 + (housing_cost + housing_change) * 0.4 + (broadband + broadband_change) * 0.3
            
            health_improvement = current_health - new_health
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Health Risk Score", f"{current_health:.1f}")
            with col2:
                st.metric("Projected Health Risk Score", f"{new_health:.1f}")
            with col3:
                st.metric("Health Improvement", f"{health_improvement:.1f}")
            
            if health_improvement > 0:
                st.markdown("""
                <div class="success-box">
                    <h4>Positive Impact Projected!</h4>
                    <p>The proposed changes are expected to improve health outcomes in the target area.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <h4>Impact Needs Attention</h4>
                    <p>The proposed changes may not provide the desired health improvements. Consider adjusting the intervention strategy.</p>
                </div>
                """, unsafe_allow_html=True)

else:
    st.info("Please load SDOH data from the sidebar to begin analysis")
    
    # Show data overview
    st.subheader("Data Overview")
    st.markdown("""
    <div class="info-box">
        <h4>SDOH Dataset Features:</h4>
        <ul>
            <li><strong>Geographic Coverage:</strong> All 50 states + DC</li>
            <li><strong>Data Level:</strong> Census tract (neighborhood-level)</li>
            <li><strong>Time Period:</strong> 2017-2021 (5-year ACS data)</li>
            <li><strong>Key Measures:</strong> 9 social determinants of health factors</li>
        </ul>
    </div>
    
    <div class="info-box">
        <h4>Available SDOH Factors:</h4>
        <ol>
            <li>Crowding among housing units</li>
            <li>Housing cost burden among households</li>
            <li>No broadband internet subscription</li>
            <li>No high school diploma among adults 25+</li>
            <li>Persons aged 65+ years</li>
            <li>Persons living below 150% poverty level</li>
            <li>Persons of racial/ethnic minority status</li>
            <li>Single-parent households</li>
            <li>Unemployment among people 16+ in labor force</li>
        </ol>
    </div>
    """, unsafe_allow_html=True) 