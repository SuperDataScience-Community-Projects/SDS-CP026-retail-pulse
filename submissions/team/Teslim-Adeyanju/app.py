#!/usr/bin/env python3
"""
Mobile Shop Customer Prediction - Model Training Script
Run this locally to train and save the model files
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier
from joblib import dump
import os
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the customer data"""
    print("📊 Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv('data.csv')
    print(f"📈 Dataset shape: {df.shape}")
    
    # Rename columns
    df.rename(columns={
        'Cus.ID':                          'customer_id',
        'Date':                            'visit_date',
        'Cus. Location':                   'customer_location',
        'Age':                             'age',
        'Gender':                          'gender',
        'Mobile Name':                     'mobile_name',
        'Sell Price':                      'sell_price',
        'Does he/she Come from Facebook Page?': 'came_from_facebook',
        'Does he/she Followed Our Page?':  'follows_facebook_page',
        'Did he/she buy any mobile before?':   'previous_customer',
        'Did he/she hear of our shop before?':'heard_about_shop'
    }, inplace=True)

    # Drop irrelevant columns 
    df.drop(columns=['customer_id','visit_date'], inplace=True)

    # Clean categorical data
    def clean_column_content(df, column_name):
        df[column_name] = df[column_name].astype(str)
        df[column_name] = df[column_name].str.lower()
        df[column_name] = df[column_name].str.replace(' ', '_')
        return df

    for col in df.select_dtypes(include=['object']).columns:
        df = clean_column_content(df, col)

    # Create engagement score
    df['engagement_score'] = (
        df['came_from_facebook'].map({'yes': 1, 'no': 0}).fillna(0) +
        df['follows_facebook_page'].map({'yes': 1, 'no': 0}).fillna(0) +
        df['heard_about_shop'].map({'yes': 1, 'no': 0}).fillna(0)
    )

    # Create engagement level
    df['engagement_level'] = pd.cut(
        df['engagement_score'],
        bins=[-1, 1, 2, 3],
        labels=['Low', 'Medium', 'High']
    )

    # Create price categories
    min_price = df['sell_price'].min()
    max_price = df['sell_price'].max()
    print(f"💰 Price range: ${min_price:,.0f} - ${max_price:,.0f}")

    df['price_category'] = pd.cut(
        df['sell_price'],
        bins=[min_price-1, 15000, 30000, max_price],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )

    return df

def prepare_features(df):
    """Prepare features for model training"""
    print("🔧 Preparing features...")
    
    # Target variable
    y = df['previous_customer'].map({'yes': 1, 'no': 0})
    
    # Check target distribution
    target_distribution = y.value_counts()
    print(f"🎯 Target distribution:")
    print(f"   New customers (0): {target_distribution[0]} ({target_distribution[0]/len(y)*100:.1f}%)")
    print(f"   Returning customers (1): {target_distribution[1]} ({target_distribution[1]/len(y)*100:.1f}%)")

    # Feature preprocessing
    binary_columns = ['gender', 'came_from_facebook', 'follows_facebook_page', 'heard_about_shop']
    categorical_columns = ['customer_location', 'mobile_name', 'engagement_level', 'price_category']

    # Label encode binary columns
    label_encoder = LabelEncoder()
    for col in binary_columns:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col].astype(str))

    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True, dtype='int64')

    # Prepare features (exclude target variable)
    feature_columns = [col for col in df_encoded.columns if col != 'previous_customer']
    X = df_encoded[feature_columns]
    
    print(f"🔧 Number of features: {len(feature_columns)}")
    
    return X, y, feature_columns, target_distribution

def train_model(X, y):
    """Train the XGBoost model"""
    print("\n🤖 Training XGBoost Model...")
    print("-" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📈 Training set: {X_train.shape[0]} samples")
    print(f"📉 Test set: {X_test.shape[0]} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train_scaled, y_train)
    print("✅ Model training completed!")

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    print(f"\n📊 Model Performance:")
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"📈 AUC Score: {auc_score:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, scaler, accuracy, auc_score

def save_model_files(model, scaler, feature_columns, target_distribution, accuracy, auc_score):
    """Save model files and metadata"""
    print("\n💾 Saving model files...")
    print("-" * 40)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    
    # Save main files (for deployment)
    dump(model, 'xgboost_model.joblib')
    dump(scaler, 'scaler.joblib')
    
    # Save backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump(model, f'models/xgboost_model_{timestamp}.joblib')
    dump(scaler, f'models/scaler_{timestamp}.joblib')
    
    # Save model metadata
    model_info = {
        'model_type': 'XGBoost Classifier',
        'accuracy': float(accuracy),
        'auc_score': float(auc_score),
        'training_date': datetime.now().isoformat(),
        'features_used': feature_columns,
        'n_features': len(feature_columns),
        'target_distribution': {
            'new_customers': int(target_distribution[0]),
            'returning_customers': int(target_distribution[1])
        },
        'model_parameters': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6
        }
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Model files saved successfully!")
    print(f"📁 Main files:")
    print(f"   - xgboost_model.joblib")
    print(f"   - scaler.joblib") 
    print(f"   - model_info.json")
    
    return True

def main():
    """Main training pipeline"""
    print("🚀 Starting Mobile Shop Customer Prediction Model Training...")
    print("=" * 60)
    
    try:
        # Check if data file exists
        if not os.path.exists('data.csv'):
            print("❌ Error: data.csv not found!")
            print("📁 Please ensure data.csv is in the same directory as this script")
            return False
        
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Prepare features
        X, y, feature_columns, target_distribution = prepare_features(df)
        
        # Train model
        model, scaler, accuracy, auc_score = train_model(X, y)
        
        # Save model files
        save_model_files(model, scaler, feature_columns, target_distribution, accuracy, auc_score)
        
        print(f"\n🎉 Training completed successfully!")
        print(f"🎯 Final Model Performance:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - AUC Score: {auc_score:.4f}")
        print(f"\n🚀 Ready for Streamlit deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔥 Mobile Shop Customer Prediction - Training Script")
    print("📋 This script will train your model and save files for deployment")
    print("=" * 60)
    
    success = main()
    
    if success:
        print("\n" + "="*60)
        print("✅ TRAINING SUCCESSFUL!")
        print("📋 Next steps:")
        print("   1. Test locally: streamlit run streamlit_app.py")
        print("   2. Upload files to GitHub")
        print("   3. Deploy on Render")
        print("="*60)
    else:
        print("\n❌ Training failed. Please check the errors above.")