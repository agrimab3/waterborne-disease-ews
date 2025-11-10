"""
Waterborne Disease Early Warning System - Data Generator
Generates realistic synthetic environmental and disease data for training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(n_samples=3000, seed=42):
    """
    Generate synthetic environmental and disease outbreak data
    
    Parameters:
    -----------
    n_samples : int
        Number of data points to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Synthetic dataset with environmental features and outbreak labels
    """
    np.random.seed(seed)
    
    # Generate date range (3 years of daily data)
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    # Generate geographical locations (5 districts)
    districts = ['District_A', 'District_B', 'District_C', 'District_D', 'District_E']
    locations = np.random.choice(districts, n_samples)
    
    # Meteorological Features
    # Temperature (15-40°C, with seasonal variation)
    base_temp = 25 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    mean_temp = base_temp + np.random.normal(0, 3, n_samples)
    
    # Precipitation (0-200mm, with rainy season)
    seasonal_rain = 50 + 40 * np.sin(2 * np.pi * np.arange(n_samples) / 365)
    precipitation = np.maximum(0, seasonal_rain + np.random.gamma(2, 15, n_samples))
    
    # Humidity (40-95%)
    humidity = 60 + 15 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 8, n_samples)
    humidity = np.clip(humidity, 40, 95)
    
    # Hydrological Features
    # Water turbidity (0-20 NTU)
    turbidity = np.maximum(0, 3 + 0.05 * precipitation + np.random.gamma(2, 1.5, n_samples))
    
    # River water level (0-10 meters)
    water_level = 3 + 0.02 * precipitation + np.random.normal(0, 0.5, n_samples)
    water_level = np.clip(water_level, 0, 10)
    
    # Groundwater level (5-20 meters depth)
    groundwater_level = 12 - 0.01 * precipitation + np.random.normal(0, 1, n_samples)
    groundwater_level = np.clip(groundwater_level, 5, 20)
    
    # Sanitation infrastructure quality index (0-100)
    district_quality = {
        'District_A': 75, 'District_B': 65, 'District_C': 55, 
        'District_D': 80, 'District_E': 60
    }
    sanitation_index = np.array([district_quality[loc] + np.random.normal(0, 5) for loc in locations])
    sanitation_index = np.clip(sanitation_index, 0, 100)
    
    # Population density (people per sq km)
    district_density = {
        'District_A': 5000, 'District_B': 3500, 'District_C': 7000,
        'District_D': 2500, 'District_E': 4500
    }
    population_density = np.array([district_density[loc] + np.random.normal(0, 500) for loc in locations])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'District': locations,
        'Mean_Temperature': np.round(mean_temp, 2),
        'Precipitation': np.round(precipitation, 2),
        'Humidity': np.round(humidity, 2),
        'Turbidity': np.round(turbidity, 2),
        'Water_Level': np.round(water_level, 2),
        'Groundwater_Level': np.round(groundwater_level, 2),
        'Sanitation_Index': np.round(sanitation_index, 2),
        'Population_Density': np.round(population_density, 0)
    })
    
    # Feature Engineering: Rolling averages (key predictors)
    df['Precipitation_7day_Avg'] = df.groupby('District')['Precipitation'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df['Precipitation_14day_Avg'] = df.groupby('District')['Precipitation'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean()
    )
    df['Turbidity_7day_Avg'] = df.groupby('District')['Turbidity'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Generate Outbreak Risk Labels
    # Complex relationship: High rain + High turbidity + Low sanitation = High risk
    risk_score = (
        0.3 * (df['Precipitation_7day_Avg'] / 100) +  # Heavy recent rainfall
        0.25 * (df['Turbidity'] / 10) +  # Water contamination
        0.2 * (1 - df['Sanitation_Index'] / 100) +  # Poor sanitation
        0.15 * (df['Mean_Temperature'] / 35) +  # Warm temperature (bacteria growth)
        0.1 * (df['Population_Density'] / 7000)  # High density
    )
    
    # Add some randomness to make it realistic
    risk_score = risk_score + np.random.normal(0, 0.1, n_samples)
    
    # Convert continuous risk score to categorical labels
    df['Risk_Score'] = np.clip(risk_score, 0, 1)
    
    # Create risk levels: 0=Low, 1=Medium, 2=High
    df['Outbreak_Risk_Level'] = pd.cut(
        df['Risk_Score'],
        bins=[-np.inf, 0.35, 0.65, np.inf],
        labels=[0, 1, 2]
    ).astype(int)
    
    # Simulate actual outbreak occurrences (binary)
    outbreak_proba = df['Risk_Score'] ** 2  # Non-linear relationship
    df['Outbreak_Occurred'] = (np.random.random(n_samples) < outbreak_proba).astype(int)
    
    # Number of reported cases
    df['Cases_Reported'] = np.where(
        df['Outbreak_Occurred'] == 1,
        np.random.poisson(lam=20 * df['Risk_Score'], size=n_samples),
        0
    )
    
    return df


if __name__ == "__main__":
    print("🌊 Generating Waterborne Disease Early Warning System Dataset...")
    print("=" * 70)
    
    # Generate data
    data = generate_synthetic_data(n_samples=3000, seed=42)
    
    # Save to CSV
    output_file = 'historical_health_environmental_data.csv'
    data.to_csv(output_file, index=False)
    
    print(f"✅ Dataset generated successfully!")
    print(f"📊 Saved to: {output_file}")
    print(f"📈 Total records: {len(data)}")
    print(f"\n🔍 Dataset Preview:")
    print(data.head())
    print(f"\n📉 Outbreak Risk Distribution:")
    print(data['Outbreak_Risk_Level'].value_counts().sort_index())
    print(f"\n🚨 Total Outbreaks: {data['Outbreak_Occurred'].sum()}")
    print(f"📊 Total Cases: {data['Cases_Reported'].sum()}")
