"""
Waterborne Disease Early Warning System - Interactive Dashboard
Real-time visualization and monitoring interface using Streamlit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Waterborne Disease EWS",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-medium {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model"""
    return joblib.load('ews_model.pkl')


@st.cache_data
def load_data():
    """Load historical data"""
    return pd.read_csv('historical_health_environmental_data.csv')


def predict_risk(model, data_point):
    """Make prediction for a single data point"""
    probabilities = model.predict_proba(data_point)[0]
    prediction = model.predict(data_point)[0]
    
    risk_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
    
    return {
        'risk_level': prediction,
        'risk_label': risk_labels[prediction],
        'prob_low': probabilities[0],
        'prob_medium': probabilities[1],
        'prob_high': probabilities[2]
    }


def main():
    # Header
    st.markdown('<p class="main-header">🌊 Waterborne Disease Early Warning System</p>', 
                unsafe_allow_html=True)
    st.markdown("### AI-Powered Predictive Healthcare for SDG 3, 6 & 2")
    st.markdown("---")
    
    # Load model and data
    try:
        model = load_model()
        historical_data = load_data()
    except FileNotFoundError:
        st.error("⚠️ Model or data files not found. Please run data_generator.py and train_model.py first.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Control Panel")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Risk Predictor", "📈 Historical Analysis", "ℹ️ About"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard(model, historical_data)
    elif page == "🔍 Risk Predictor":
        show_predictor(model)
    elif page == "📈 Historical Analysis":
        show_analysis(historical_data)
    else:
        show_about()


def show_dashboard(model, data):
    """Main dashboard view"""
    st.header("Real-Time Regional Monitoring")
    
    # Simulate current conditions for 5 districts
    districts = ['District_A', 'District_B', 'District_C', 'District_D', 'District_E']
    
    # Generate simulated current data
    current_conditions = []
    for district in districts:
        condition = {
            'District': district,
            'Mean_Temperature': np.random.uniform(22, 35),
            'Precipitation': np.random.gamma(2, 20),
            'Humidity': np.random.uniform(50, 90),
            'Turbidity': np.random.gamma(2, 2),
            'Water_Level': np.random.uniform(2, 8),
            'Groundwater_Level': np.random.uniform(8, 16),
            'Sanitation_Index': np.random.uniform(45, 85),
            'Population_Density': np.random.uniform(2000, 7000),
        }
        # Add rolling averages (simulated)
        condition['Precipitation_7day_Avg'] = condition['Precipitation'] * 0.8
        condition['Precipitation_14day_Avg'] = condition['Precipitation'] * 0.7
        condition['Turbidity_7day_Avg'] = condition['Turbidity'] * 0.9
        
        current_conditions.append(condition)
    
    current_df = pd.DataFrame(current_conditions)
    
    # Make predictions for all districts
    feature_columns = [
        'Mean_Temperature', 'Precipitation', 'Humidity', 'Turbidity',
        'Water_Level', 'Groundwater_Level', 'Sanitation_Index',
        'Population_Density', 'Precipitation_7day_Avg',
        'Precipitation_14day_Avg', 'Turbidity_7day_Avg'
    ]
    
    predictions = []
    for idx, row in current_df.iterrows():
        pred = predict_risk(model, pd.DataFrame([row[feature_columns]]))
        predictions.append(pred)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk_count = sum(1 for p in predictions if p['risk_level'] == 2)
    medium_risk_count = sum(1 for p in predictions if p['risk_level'] == 1)
    low_risk_count = sum(1 for p in predictions if p['risk_level'] == 0)
    avg_high_prob = np.mean([p['prob_high'] for p in predictions]) * 100
    
    with col1:
        st.metric("🚨 High Risk Areas", high_risk_count, help="Districts with high outbreak risk")
    with col2:
        st.metric("⚠️ Medium Risk Areas", medium_risk_count, help="Districts with elevated risk")
    with col3:
        st.metric("✅ Low Risk Areas", low_risk_count, help="Districts with low risk")
    with col4:
        st.metric("📊 Avg High Risk Prob", f"{avg_high_prob:.1f}%", help="Average probability across all districts")
    
    st.markdown("---")
    
    # Regional risk map
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Regional Risk Map")
        
        # Create risk level visualization
        map_data = current_df.copy()
        map_data['Risk_Level'] = [p['risk_label'] for p in predictions]
        map_data['Risk_Score'] = [p['prob_high'] * 100 for p in predictions]
        
        # Create color mapping
        color_map = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
        map_data['Color'] = map_data['Risk_Level'].map(color_map)
        
        fig = px.bar(
            map_data,
            x='District',
            y='Risk_Score',
            color='Risk_Level',
            color_discrete_map=color_map,
            title='Outbreak Risk by District',
            labels={'Risk_Score': 'High Risk Probability (%)'},
            text='Risk_Score'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Active Alerts")
        
        for i, (district, pred) in enumerate(zip(districts, predictions)):
            if pred['risk_level'] == 2:
                st.markdown(f"""
                <div class="alert-high">
                    <strong>🚨 {district}</strong><br>
                    High Risk: {pred['prob_high']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
            elif pred['risk_level'] == 1:
                st.markdown(f"""
                <div class="alert-medium">
                    <strong>⚠️ {district}</strong><br>
                    Medium Risk: {pred['prob_high']*100:.1f}%
                </div>
                """, unsafe_allow_html=True)
    
    # Detailed district view
    st.markdown("---")
    st.subheader("🔍 Detailed District Information")
    
    selected_district = st.selectbox("Select District", districts)
    district_idx = districts.index(selected_district)
    district_data = current_df.iloc[district_idx]
    district_pred = predictions[district_idx]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🌡️ Temperature", f"{district_data['Mean_Temperature']:.1f}°C")
        st.metric("💧 Precipitation", f"{district_data['Precipitation']:.1f} mm")
        st.metric("💨 Humidity", f"{district_data['Humidity']:.1f}%")
    
    with col2:
        st.metric("🌊 Turbidity", f"{district_data['Turbidity']:.2f} NTU")
        st.metric("📏 Water Level", f"{district_data['Water_Level']:.2f} m")
        st.metric("🚰 Sanitation Index", f"{district_data['Sanitation_Index']:.0f}/100")
    
    with col3:
        st.metric("👥 Population Density", f"{district_data['Population_Density']:.0f}/km²")
        st.metric("📊 7-Day Avg Rain", f"{district_data['Precipitation_7day_Avg']:.1f} mm")
        st.metric("🎯 Risk Level", district_pred['risk_label'])


def show_predictor(model):
    """Interactive risk prediction tool"""
    st.header("🔮 Custom Risk Prediction")
    st.write("Enter environmental parameters to predict outbreak risk")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Meteorological Data")
        temp = st.slider("Mean Temperature (°C)", 15.0, 40.0, 28.0)
        precip = st.slider("Daily Precipitation (mm)", 0.0, 200.0, 50.0)
        humidity = st.slider("Relative Humidity (%)", 40.0, 95.0, 70.0)
        precip_7day = st.slider("7-Day Avg Precipitation (mm)", 0.0, 150.0, 40.0)
        precip_14day = st.slider("14-Day Avg Precipitation (mm)", 0.0, 120.0, 35.0)
    
    with col2:
        st.subheader("Environmental & Social Data")
        turbidity = st.slider("Water Turbidity (NTU)", 0.0, 20.0, 5.0)
        water_level = st.slider("River Water Level (m)", 0.0, 10.0, 4.0)
        groundwater = st.slider("Groundwater Level (m depth)", 5.0, 20.0, 12.0)
        sanitation = st.slider("Sanitation Index (0-100)", 0.0, 100.0, 65.0)
        population = st.slider("Population Density (/km²)", 1000.0, 8000.0, 4000.0)
        turbidity_7day = st.slider("7-Day Avg Turbidity (NTU)", 0.0, 15.0, 4.0)
    
    if st.button("🔍 Predict Risk", type="primary"):
        # Create input data
        input_data = pd.DataFrame([{
            'Mean_Temperature': temp,
            'Precipitation': precip,
            'Humidity': humidity,
            'Turbidity': turbidity,
            'Water_Level': water_level,
            'Groundwater_Level': groundwater,
            'Sanitation_Index': sanitation,
            'Population_Density': population,
            'Precipitation_7day_Avg': precip_7day,
            'Precipitation_14day_Avg': precip_14day,
            'Turbidity_7day_Avg': turbidity_7day
        }])
        
        # Make prediction
        pred = predict_risk(model, input_data)
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        # Display risk level with color coding
        if pred['risk_level'] == 2:
            st.markdown(f"""
            <div class="alert-high">
                <h3>🚨 HIGH RISK</h3>
                <p>Probability of outbreak: {pred['prob_high']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        elif pred['risk_level'] == 1:
            st.markdown(f"""
            <div class="alert-medium">
                <h3>⚠️ MEDIUM RISK</h3>
                <p>Probability of outbreak: {pred['prob_high']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-low">
                <h3>✅ LOW RISK</h3>
                <p>Probability of outbreak: {pred['prob_high']*100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability breakdown
        col1, col2, col3 = st.columns(3)
        col1.metric("Low Risk", f"{pred['prob_low']*100:.1f}%")
        col2.metric("Medium Risk", f"{pred['prob_medium']*100:.1f}%")
        col3.metric("High Risk", f"{pred['prob_high']*100:.1f}%")
        
        # Visualization
        fig = go.Figure(data=[
            go.Bar(
                x=['Low', 'Medium', 'High'],
                y=[pred['prob_low']*100, pred['prob_medium']*100, pred['prob_high']*100],
                marker_color=['green', 'orange', 'red']
            )
        ])
        fig.update_layout(
            title="Risk Probability Distribution",
            xaxis_title="Risk Level",
            yaxis_title="Probability (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def show_analysis(data):
    """Historical data analysis"""
    st.header("📈 Historical Outbreak Analysis")
    
    # Convert date column
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Time series of outbreaks
    st.subheader("📅 Outbreak Trends Over Time")
    
    daily_risk = data.groupby('Date')['Outbreak_Risk_Level'].mean().reset_index()
    
    fig = px.line(
        daily_risk,
        x='Date',
        y='Outbreak_Risk_Level',
        title='Average Risk Level Over Time',
        labels={'Outbreak_Risk_Level': 'Risk Level'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Risk distribution by district
    st.subheader("🗺️ Risk Distribution by District")
    
    district_risk = data.groupby(['District', 'Outbreak_Risk_Level']).size().reset_index(name='Count')
    
    fig = px.bar(
        district_risk,
        x='District',
        y='Count',
        color='Outbreak_Risk_Level',
        title='Risk Level Distribution by District',
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key statistics
    st.subheader("📊 Key Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    total_outbreaks = data['Outbreak_Occurred'].sum()
    total_cases = data['Cases_Reported'].sum()
    high_risk_days = (data['Outbreak_Risk_Level'] == 2).sum()
    
    col1.metric("Total Outbreaks", f"{total_outbreaks:,}")
    col2.metric("Total Cases", f"{total_cases:,}")
    col3.metric("High Risk Days", f"{high_risk_days:,}")


def show_about():
    """About page"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ### 🌊 AI-Powered Waterborne Disease Early Warning System
    
    **Mission:** Transform healthcare from reactive to proactive by predicting waterborne disease outbreaks
    before they occur.
    
    #### 🎯 UN Sustainable Development Goals
    
    - **SDG 3: Good Health & Well-being** - Prevents illness and reduces hospital burden
    - **SDG 6: Clean Water & Sanitation** - Improves water safety through real-time warnings
    - **SDG 2: Zero Hunger** - Prevents nutrient malabsorption from waterborne diseases
    
    #### 🤖 Technology Stack
    
    - **Machine Learning:** Random Forest Classifier with 200 decision trees
    - **Data Processing:** Pandas, NumPy for environmental data analysis
    - **Visualization:** Plotly, Streamlit for interactive dashboards
    - **Prediction Features:** 11 environmental and socio-economic indicators
    
    #### 📊 Model Performance
    
    - High accuracy in predicting outbreak risk levels
    - Real-time monitoring of multiple geographical regions
    - Evidence-based alert system for public health officials
    
    #### 💡 Key Features
    
    1. **Predictive Analytics:** 7-day advance warning of potential outbreaks
    2. **Multi-Region Monitoring:** Simultaneous tracking of multiple districts
    3. **Intelligent Alerts:** Risk-stratified notification system
    4. **Interactive Dashboard:** Real-time visualization and custom predictions
    
    ---
    
    **Developed for:** Congressional App Challenge / University Applications  
    **Focus:** Computational Neuroscience meets Public Health Technology
    """)


if __name__ == "__main__":
    main()
