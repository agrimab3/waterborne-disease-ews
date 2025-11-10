"""
Waterborne Disease Early Warning System - Model Training
Trains a Random Forest classifier to predict outbreak risk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_auc_score, roc_curve
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_and_prepare_data(filepath='historical_health_environmental_data.csv'):
    """Load and prepare data for model training"""
    print("📂 Loading data...")
    data = pd.read_csv(filepath)
    
    # Convert date to datetime
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Select features for the model
    feature_columns = [
        'Mean_Temperature', 'Precipitation', 'Humidity', 'Turbidity',
        'Water_Level', 'Groundwater_Level', 'Sanitation_Index',
        'Population_Density', 'Precipitation_7day_Avg', 
        'Precipitation_14day_Avg', 'Turbidity_7day_Avg'
    ]
    
    # Prepare features (X) and target (Y)
    X = data[feature_columns]
    Y = data['Outbreak_Risk_Level']  # 0=Low, 1=Medium, 2=High
    
    print(f"✅ Data loaded: {len(data)} records")
    print(f"📊 Features: {len(feature_columns)}")
    print(f"🎯 Target distribution:")
    print(Y.value_counts().sort_index())
    
    return X, Y, feature_columns, data


def train_model(X, Y):
    """Train Random Forest classifier"""
    print("\n🤖 Training Random Forest model...")
    
    # Split data into train and test sets (80/20 split)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )
    
    print(f"📊 Training set: {len(X_train)} samples")
    print(f"📊 Test set: {len(X_test)} samples")
    
    # Initialize Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=200,  # Number of trees
        max_depth=15,       # Maximum depth of trees
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1           # Use all CPU cores
    )
    
    # Train the model
    print("⏳ Training in progress...")
    model.fit(X_train, Y_train)
    print("✅ Model training complete!")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')
    print(f"\n📈 Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    return model, X_train, X_test, Y_train, Y_test


def evaluate_model(model, X_test, Y_test, feature_columns):
    """Evaluate model performance"""
    print("\n📊 Model Evaluation")
    print("=" * 70)
    
    # Make predictions
    Y_pred = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    report = classification_report(
        Y_test, Y_pred,
        target_names=['Low Risk', 'Medium Risk', 'High Risk'],
        digits=4
    )
    print(report)
    
    # Feature importance
    print("\n🔍 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']:.<35} {row['Importance']:.4f}")
    
    return Y_pred, Y_proba, feature_importance


def create_visualizations(model, X_test, Y_test, Y_pred, feature_importance):
    """Create comprehensive visualizations"""
    print("\n📊 Creating visualizations...")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(Y_test, Y_pred)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Low', 'Medium', 'High'],
        yticklabels=['Low', 'Medium', 'High']
    )
    plt.title('Confusion Matrix - Outbreak Risk Prediction', fontsize=16, fontweight='bold')
    plt.ylabel('True Risk Level', fontsize=12)
    plt.xlabel('Predicted Risk Level', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: confusion_matrix.png")
    
    # 2. Feature Importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    bars = plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Top 10 Most Important Predictive Features', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Color bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: feature_importance.png")
    
    # 3. Model Performance Summary
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy by class
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(Y_test, Y_pred)
    
    x = ['Low Risk', 'Medium Risk', 'High Risk']
    width = 0.25
    x_pos = np.arange(len(x))
    
    axes[0].bar(x_pos - width, precision, width, label='Precision', color='skyblue')
    axes[0].bar(x_pos, recall, width, label='Recall', color='lightcoral')
    axes[0].bar(x_pos + width, f1, width, label='F1-Score', color='lightgreen')
    axes[0].set_xlabel('Risk Level', fontsize=12)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Model Performance by Risk Level', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x)
    axes[0].legend()
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Prediction distribution
    unique, counts = np.unique(Y_pred, return_counts=True)
    colors_pie = ['#90EE90', '#FFD700', '#FF6B6B']
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    
    # Only use labels and colors for classes that exist in predictions
    used_labels = [labels[i] for i in unique]
    used_colors = [colors_pie[i] for i in unique]
    
    axes[1].pie(
        counts, labels=used_labels,
        autopct='%1.1f%%', colors=used_colors, startangle=90
    )
    axes[1].set_title('Distribution of Predictions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved: model_performance.png")
    
    plt.close('all')


def save_model(model, filename='ews_model.pkl'):
    """Save trained model to disk"""
    joblib.dump(model, filename)
    print(f"\n💾 Model saved to: {filename}")


def main():
    """Main training pipeline"""
    print("🌊 WATERBORNE DISEASE EARLY WARNING SYSTEM")
    print("=" * 70)
    print("📚 Machine Learning Model Training Pipeline\n")
    
    # Step 1: Load data
    X, Y, feature_columns, data = load_and_prepare_data()
    
    # Step 2: Train model
    model, X_train, X_test, Y_train, Y_test = train_model(X, Y)
    
    # Step 3: Evaluate model
    Y_pred, Y_proba, feature_importance = evaluate_model(
        model, X_test, Y_test, feature_columns
    )
    
    # Step 4: Create visualizations
    create_visualizations(model, X_test, Y_test, Y_pred, feature_importance)
    
    # Step 5: Save model
    save_model(model)
    
    print("\n" + "=" * 70)
    print("✅ Training pipeline completed successfully!")
    print("🎉 Your model is ready to predict waterborne disease outbreaks!")
    
    return model


if __name__ == "__main__":
    model = main()
