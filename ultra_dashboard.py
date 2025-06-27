import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
from utils.data_processor import SDOHDataProcessor
from ml_pipeline import SDOHMLPipeline
import config
import time
from streamlit_option_menu import option_menu

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Digital Twin Cities Ultra Dashboard",
    page_icon="🌃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR GLASSMORPHISM & NAV ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #181c24 0%, #232946 100%) !important;
}
#MainMenu, header, footer {visibility: hidden;}
.hero-metric {
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: 2px;
    background: linear-gradient(90deg, #6EE7B7, #3B82F6, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.hero-desc {
    font-size: 1.2rem;
    color: #cbd5e1;
    margin-bottom: 1.5rem;
}
.glass-card {
    background: rgba(36, 41, 54, 0.7);
    border-radius: 18px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1.5px solid rgba(255,255,255,0.08);
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    color: #e0e7ef;
}
.floating-panel {
    position: absolute;
    top: 80px;
    right: 4vw;
    min-width: 320px;
    max-width: 400px;
    background: rgba(36, 41, 54, 0.92);
    border-radius: 18px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
    padding: 1.5rem 2rem;
    color: #e0e7ef;
    z-index: 100;
    border: 1.5px solid rgba(255,255,255,0.08);
}
@media (max-width: 900px) {
    .floating-panel { position: static; margin: 2rem auto; }
}
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (MINIMAL) ---
st.sidebar.markdown("## Controls")
st.sidebar.markdown("---")
if 'sdoh_data_loaded' not in st.session_state:
    st.session_state.sdoh_data_loaded = False
if 'processor' not in st.session_state:
    st.session_state.processor = None

if not st.session_state.sdoh_data_loaded:
    if st.sidebar.button("Load SDOH Data"):
        with st.spinner("Loading data..."):
            processor = SDOHDataProcessor("Non-Medical_Factor_Measures_for_Census_Tract__ACS_2017-2021_20250626.csv")
            processor.load_data(sample_size=50000)
            processor.pivot_data()
            processor.handle_missing_values(strategy='drop')
            processor.encode_categorical_features()
            processor.create_target_variable(target_type='health_index')
            st.session_state.processor = processor
            st.session_state.sdoh_data_loaded = True
        st.sidebar.success("SDOH data loaded")
else:
    st.sidebar.success("SDOH data loaded")
    if st.sidebar.button("Reset Data"):
        st.session_state.sdoh_data_loaded = False
        st.session_state.processor = None
        st.experimental_rerun()
    states = ['All States'] + sorted(st.session_state.processor.processed_df['StateAbbr'].unique().tolist())
    selected_state = st.sidebar.selectbox("Select State", states)
    if selected_state == 'All States':
        selected_state = None
    if st.sidebar.button("Run ML Pipeline"):
        with st.spinner("Running ML pipeline..."):
            pipeline = SDOHMLPipeline("Non-Medical_Factor_Measures_for_Census_Tract__ACS_2017-2021_20250626.csv")
            pipeline.data_processor = st.session_state.processor
            pipeline.prepare_models()
            pipeline.train_all_models(target_col='health_index')
            st.session_state.ml_pipeline = pipeline
        st.sidebar.success("ML pipeline complete!")

# --- TOP NAVIGATION BAR (streamlit-option-menu) ---
with st.container():
    selected_tab = option_menu(
        menu_title=None,
        options=["Overview", "Map", "Impact"],
        icons=["house", "map", "activity"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "rgba(24,28,36,0.95)"},
            "icon": {"color": "#A78BFA", "font-size": "20px"},
            "nav-link": {"font-size": "18px", "font-weight": "bold", "color": "#cbd5e1", "margin": "0 12px", "border-radius": "8px"},
            "nav-link-selected": {"background": "linear-gradient(90deg, #3B82F6 60%, #A78BFA 100%)", "color": "#fff"},
        }
    )

# --- HERO SECTION (OVERVIEW) ---
if selected_tab == "Overview":
    st.markdown('<div class="glass-card" style="margin-top:2rem;text-align:center;">', unsafe_allow_html=True)
    st.markdown('<div class="hero-metric">Digital Twin Cities</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-desc">A next-generation platform for exploring, modeling, and impacting Social Determinants of Health across the US.</div>', unsafe_allow_html=True)
    if st.session_state.sdoh_data_loaded:
        df = st.session_state.processor.processed_df
        st.markdown("<hr>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Census Tracts", f"{len(df):,}")
        with col2:
            st.metric("States", f"{df['StateAbbr'].nunique()}")
        with col3:
            st.metric("Population", f"{int(df['TotalPopulation'].sum()):,}")
        with col4:
            st.metric("Avg Health Index", f"{df['health_index'].mean():.1f}")
        st.markdown("<br>", unsafe_allow_html=True)
        fig = px.histogram(df, x='health_index', nbins=30, color_discrete_sequence=['#6366f1'])
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e7ef',
            title_text='Distribution of Health Index',
            title_font_size=22
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load SDOH data to begin.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- MAP SECTION ---
if selected_tab == "Map" and st.session_state.sdoh_data_loaded:
    st.markdown('<div class="glass-card" style="margin-top:2rem;">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#fff;">Interactive SDOH Map</h2>', unsafe_allow_html=True)
    df = st.session_state.processor.processed_df
    if selected_state:
        df = df[df['StateAbbr'] == selected_state]
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=5, tiles='cartodbpositron')
    for _, row in df.iterrows():
        color = '#16a34a' if row['health_index'] <= 25 else ('#f59e0b' if row['health_index'] <= 50 else '#dc2626')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"<b>{row['CountyName']}, {row['StateAbbr']}</b><br>Health Index: {row['health_index']:.1f}"
        ).add_to(m)
    map_data = st_folium(m, width=1100, height=600)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="floating-panel"><b>Click a marker for details.</b><br>Green: Good (≤25)<br>Orange: Moderate (≤50)<br>Red: Poor (>50)</div>', unsafe_allow_html=True)

# --- IMPACT SECTION ---
if selected_tab == "Impact" and st.session_state.sdoh_data_loaded:
    st.markdown('<div class="glass-card" style="margin-top:2rem;">', unsafe_allow_html=True)
    st.markdown('<h2 style="color:#fff;">Policy Impact Calculator</h2>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        crowding = st.slider("Crowding (%)", 0.0, 20.0, 5.0)
        housing = st.slider("Housing Cost Burden (%)", 0.0, 50.0, 25.0)
        broadband = st.slider("No Broadband (%)", 0.0, 30.0, 10.0)
    with col2:
        crowding_chg = st.slider("Crowding Change (%)", -5.0, 5.0, 0.0)
        housing_chg = st.slider("Housing Cost Change (%)", -10.0, 10.0, 0.0)
        broadband_chg = st.slider("Broadband Change (%)", -10.0, 10.0, 0.0)
    if st.button("Calculate Impact"):
        current = crowding * 0.3 + housing * 0.4 + broadband * 0.3
        new = (crowding + crowding_chg) * 0.3 + (housing + housing_chg) * 0.4 + (broadband + broadband_chg) * 0.3
        improvement = current - new
        color = "#16a34a" if improvement > 0 else "#f59e0b"
        icon = "✅" if improvement > 0 else "⚠️"
        st.markdown(f'''
        <div style="margin:2rem auto;max-width:420px;padding:2.5rem 2rem;background:rgba(36,41,54,0.92);border-radius:22px;box-shadow:0 4px 32px #0005;text-align:center;">
            <div style="font-size:3rem;margin-bottom:0.5rem;color:{color};">{icon}</div>
            <h3 style="color:#fff;margin-bottom:1.2rem;">Impact Assessment</h3>
            <div style="font-size:1.15rem;color:#cbd5e1;margin-bottom:1.2rem;">
                <b>Current Health Risk:</b> {current:.1f}<br>
                <b>Projected Health Risk:</b> {new:.1f}<br>
                <b>Improvement:</b> <span style="color:{color};font-weight:700;">{improvement:.1f}</span>
            </div>
            <div style="margin-top:1.5rem;font-size:1.1rem;">
                {'<span style="color:#16a34a;font-weight:600;">Positive impact projected!</span>' if improvement>0 else '<span style="color:#f59e0b;font-weight:600;">Consider adjusting your intervention.</span>'}
            </div>
        </div>
        ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- EMPTY STATES ---
if not st.session_state.sdoh_data_loaded:
    st.markdown('<div class="glass-card" style="margin-top:2rem;text-align:center;">', unsafe_allow_html=True)
    st.markdown('<div class="hero-metric">Digital Twin Cities</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-desc">Load SDOH data to begin exploring the most beautiful dashboard in the world.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True) 