"""
Waterborne Disease Early Warning System - Alert System
Real-time monitoring and alert generation for high-risk predictions
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class WaterborneEWSAlertSystem:
    """Alert system for waterborne disease outbreak prediction"""
    
    def __init__(self, model_path='ews_model.pkl', threshold_high=0.70, threshold_medium=0.40):
        """
        Initialize alert system
        
        Parameters:
        -----------
        model_path : str
            Path to trained model file
        threshold_high : float
            Probability threshold for high risk alert (0-1)
        threshold_medium : float
            Probability threshold for medium risk alert (0-1)
        """
        print("🚨 Initializing Early Warning Alert System...")
        self.model = joblib.load(model_path)
        self.threshold_high = threshold_high
        self.threshold_medium = threshold_medium
        print(f"✅ Model loaded from {model_path}")
        print(f"⚠️  High Risk Threshold: {threshold_high*100:.0f}%")
        print(f"⚠️  Medium Risk Threshold: {threshold_medium*100:.0f}%")
    
    def predict_risk(self, data_point):
        """
        Predict outbreak risk for a single data point
        
        Parameters:
        -----------
        data_point : pd.DataFrame or dict
            Environmental data for prediction
        
        Returns:
        --------
        dict : Prediction results with risk level and probabilities
        """
        # Convert dict to DataFrame if necessary
        if isinstance(data_point, dict):
            data_point = pd.DataFrame([data_point])
        
        # Make prediction
        prediction = self.model.predict(data_point)[0]
        probabilities = self.model.predict_proba(data_point)[0]
        
        # Get the highest probability
        max_prob = probabilities[prediction]
        
        # Determine risk category
        risk_labels = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}
        risk_label = risk_labels[prediction]
        
        return {
            'risk_level': prediction,
            'risk_label': risk_label,
            'confidence': max_prob,
            'probability_low': probabilities[0],
            'probability_medium': probabilities[1],
            'probability_high': probabilities[2]
        }
    
    def check_for_alert(self, data_point, location="Unknown"):
        """
        Check if an alert should be triggered
        
        Parameters:
        -----------
        data_point : pd.DataFrame or dict
            Environmental data for prediction
        location : str
            Geographic location identifier
        
        Returns:
        --------
        dict : Alert information
        """
        # Get prediction
        prediction = self.predict_risk(data_point)
        
        # Determine alert level
        high_risk_prob = prediction['probability_high']
        
        alert = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'location': location,
            'alert_triggered': False,
            'alert_level': None,
            'risk_prediction': prediction,
            'message': None,
            'recommended_actions': []
        }
        
        # Check thresholds
        if high_risk_prob >= self.threshold_high:
            alert['alert_triggered'] = True
            alert['alert_level'] = 'HIGH'
            alert['message'] = f"🚨 URGENT ALERT: High risk of waterborne disease outbreak detected!"
            alert['recommended_actions'] = [
                "Immediately notify public health officials",
                "Issue water quality advisory to residents",
                "Increase water treatment and monitoring",
                "Prepare medical facilities for potential cases",
                "Distribute water purification supplies"
            ]
        elif high_risk_prob >= self.threshold_medium:
            alert['alert_triggered'] = True
            alert['alert_level'] = 'MEDIUM'
            alert['message'] = f"⚠️  WARNING: Elevated risk of waterborne disease outbreak"
            alert['recommended_actions'] = [
                "Alert local health department",
                "Increase water quality testing frequency",
                "Review sanitation infrastructure",
                "Prepare public health communication materials",
                "Monitor situation closely"
            ]
        else:
            alert['alert_level'] = 'LOW'
            alert['message'] = f"✅ Low risk: Continue routine monitoring"
            alert['recommended_actions'] = [
                "Maintain standard water quality protocols",
                "Continue regular surveillance"
            ]
        
        return alert
    
    def print_alert(self, alert):
        """Pretty print alert information"""
        print("\n" + "="*70)
        print(f"🌊 WATERBORNE DISEASE EARLY WARNING SYSTEM")
        print("="*70)
        print(f"📅 Timestamp: {alert['timestamp']}")
        print(f"📍 Location: {alert['location']}")
        print(f"\n{alert['message']}")
        print(f"🎯 Alert Level: {alert['alert_level']}")
        
        pred = alert['risk_prediction']
        print(f"\n📊 Risk Assessment:")
        print(f"  Predicted Risk: {pred['risk_label']}")
        print(f"  Confidence: {pred['confidence']*100:.1f}%")
        print(f"\n  Detailed Probabilities:")
        print(f"    Low Risk:    {pred['probability_low']*100:>6.2f}%")
        print(f"    Medium Risk: {pred['probability_medium']*100:>6.2f}%")
        print(f"    High Risk:   {pred['probability_high']*100:>6.2f}%")
        
        if alert['recommended_actions']:
            print(f"\n💡 Recommended Actions:")
            for i, action in enumerate(alert['recommended_actions'], 1):
                print(f"  {i}. {action}")
        
        print("="*70)
    
    def monitor_region(self, current_data):
        """
        Monitor multiple regions and generate alerts
        
        Parameters:
        -----------
        current_data : pd.DataFrame
            Current environmental data for multiple regions
        
        Returns:
        --------
        list : List of alerts for each region
        """
        alerts = []
        
        print("\n🔍 Monitoring Multiple Regions...")
        print("="*70)
        
        for idx, row in current_data.iterrows():
            location = row.get('District', f'Location_{idx}')
            
            # Prepare data point (exclude non-feature columns)
            feature_columns = [
                'Mean_Temperature', 'Precipitation', 'Humidity', 'Turbidity',
                'Water_Level', 'Groundwater_Level', 'Sanitation_Index',
                'Population_Density', 'Precipitation_7day_Avg',
                'Precipitation_14day_Avg', 'Turbidity_7day_Avg'
            ]
            
            data_point = row[feature_columns].to_dict()
            
            # Check for alert
            alert = self.check_for_alert(data_point, location)
            alerts.append(alert)
            
            # Print summary
            symbol = "🚨" if alert['alert_level'] == 'HIGH' else "⚠️ " if alert['alert_level'] == 'MEDIUM' else "✅"
            print(f"{symbol} {location}: {alert['alert_level']} Risk - "
                  f"P(High)={alert['risk_prediction']['probability_high']*100:.1f}%")
        
        print("="*70)
        
        # Summary statistics
        high_alerts = sum(1 for a in alerts if a['alert_level'] == 'HIGH')
        medium_alerts = sum(1 for a in alerts if a['alert_level'] == 'MEDIUM')
        
        print(f"\n📈 Summary: {high_alerts} HIGH alerts, {medium_alerts} MEDIUM alerts")
        
        return alerts


def simulate_real_time_scenarios():
    """Simulate different environmental scenarios and alerts"""
    print("🌊 SIMULATING REAL-TIME SCENARIOS")
    print("="*70 + "\n")
    
    # Initialize alert system
    alert_system = WaterborneEWSAlertSystem(
        model_path='ews_model.pkl',
        threshold_high=0.65,
        threshold_medium=0.40
    )
    
    # Scenario 1: High Risk - Heavy rainfall + Poor sanitation
    print("\n" + "🌧️  SCENARIO 1: Heavy Monsoon Rainfall".center(70))
    high_risk_scenario = {
        'Mean_Temperature': 32.0,
        'Precipitation': 180.0,
        'Humidity': 88.0,
        'Turbidity': 15.5,
        'Water_Level': 8.2,
        'Groundwater_Level': 8.5,
        'Sanitation_Index': 45.0,
        'Population_Density': 6500,
        'Precipitation_7day_Avg': 150.0,
        'Precipitation_14day_Avg': 120.0,
        'Turbidity_7day_Avg': 12.0
    }
    alert = alert_system.check_for_alert(high_risk_scenario, "District_C (Urban)")
    alert_system.print_alert(alert)
    
    # Scenario 2: Medium Risk - Moderate conditions
    print("\n" + "☁️  SCENARIO 2: Moderate Rainfall with Infrastructure Concerns".center(70))
    medium_risk_scenario = {
        'Mean_Temperature': 28.0,
        'Precipitation': 80.0,
        'Humidity': 72.0,
        'Turbidity': 8.5,
        'Water_Level': 5.5,
        'Groundwater_Level': 11.0,
        'Sanitation_Index': 60.0,
        'Population_Density': 4500,
        'Precipitation_7day_Avg': 70.0,
        'Precipitation_14day_Avg': 55.0,
        'Turbidity_7day_Avg': 7.0
    }
    alert = alert_system.check_for_alert(medium_risk_scenario, "District_B (Suburban)")
    alert_system.print_alert(alert)
    
    # Scenario 3: Low Risk - Good conditions
    print("\n" + "☀️  SCENARIO 3: Dry Season with Good Infrastructure".center(70))
    low_risk_scenario = {
        'Mean_Temperature': 26.0,
        'Precipitation': 15.0,
        'Humidity': 55.0,
        'Turbidity': 3.2,
        'Water_Level': 3.5,
        'Groundwater_Level': 14.0,
        'Sanitation_Index': 82.0,
        'Population_Density': 2800,
        'Precipitation_7day_Avg': 12.0,
        'Precipitation_14day_Avg': 18.0,
        'Turbidity_7day_Avg': 3.0
    }
    alert = alert_system.check_for_alert(low_risk_scenario, "District_D (Rural)")
    alert_system.print_alert(alert)


if __name__ == "__main__":
    simulate_real_time_scenarios()
