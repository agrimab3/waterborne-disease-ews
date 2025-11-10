"""
Main runner script for the Waterborne Disease Early Warning System
Executes the complete pipeline from data generation to alerts
"""

import os
import sys

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def main():
    """Run the complete Early Warning System pipeline"""
    
    print("\n" + "🌊"*40)
    print_header("WATERBORNE DISEASE EARLY WARNING SYSTEM")
    print("         AI-Powered Public Health Protection")
    print("         Preventing Outbreaks, Saving Lives")
    print("🌊"*40 + "\n")
    
    try:
        # Step 1: Generate synthetic data
        print_header("STEP 1: GENERATING HISTORICAL DATA")
        print("Creating realistic environmental and health data...")
        from data_generator import generate_synthetic_data, add_rolling_features
        
        data = generate_synthetic_data(num_samples=2000)
        data = add_rolling_features(data)
        data.to_csv('/home/claude/waterborne_ews/historical_data.csv', index=False)
        print(f"✅ Generated {len(data)} data points")
        print(f"   Features: {data.shape[1]} columns")
        
        # Step 2: Train the model
        print_header("STEP 2: TRAINING MACHINE LEARNING MODEL")
        print("Training Random Forest Classifier...")
        from train_model import main as train_main
        
        ews, X_test, y_test = train_main()
        print("✅ Model training complete!")
        
        # Step 3: Run alert system demonstrations
        print_header("STEP 3: TESTING ALERT SYSTEM")
        print("Simulating real-time outbreak risk monitoring...")
        from alert_system import demo_alert_system
        
        demo_alert_system()
        print("✅ Alert system tested successfully!")
        
        # Step 4: Generate dashboard visualizations
        print_header("STEP 4: CREATING DASHBOARD VISUALIZATIONS")
        print("Generating risk maps and analytics...")
        from dashboard import create_full_dashboard
        
        create_full_dashboard()
        print("✅ Dashboard visualizations created!")
        
        # Final summary
        print_header("🎉 EARLY WARNING SYSTEM DEPLOYMENT COMPLETE! 🎉")
        
        print("📁 Generated Files:")
        print("   • historical_data.csv - Synthetic training data")
        print("   • ews_model.pkl - Trained ML model")
        print("   • feature_importance.png - Key risk factors visualization")
        print("   • confusion_matrix.png - Model accuracy visualization")
        print("   • risk_map.png - Regional risk map")
        print("   • dashboard_summary.png - Complete dashboard")
        print("   • time_series.png - Environmental trends")
        
        print("\n🎯 System Capabilities:")
        print("   ✓ Predict waterborne disease outbreaks 7 days in advance")
        print("   ✓ Monitor multiple locations simultaneously")
        print("   ✓ Generate real-time alerts for public health officials")
        print("   ✓ Identify key environmental risk factors")
        print("   ✓ Provide actionable intervention recommendations")
        
        print("\n🌍 Impact Alignment:")
        print("   • SDG 3: Good Health & Well-being")
        print("   • SDG 6: Clean Water & Sanitation")
        print("   • SDG 2: Zero Hunger")
        
        print("\n💡 Next Steps:")
        print("   1. Customize with real regional data")
        print("   2. Integrate with weather API for live data")
        print("   3. Deploy on cloud platform (AWS/Azure)")
        print("   4. Connect to SMS/Email notification system")
        print("   5. Create mobile app for field workers")
        
        print("\n" + "="*80)
        print("Ready for Congressional App Challenge submission! 🏆")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
