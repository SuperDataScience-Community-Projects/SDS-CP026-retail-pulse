#!/usr/bin/env python3
"""
Mobile Shop Customer Prediction - Streamlit Web Interface (Frontend Only)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="📱 Mobile Shop Customer Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .prediction-result {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .returning-customer {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .new-customer {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }
    
    .info-box {
        background-color: #000000;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #333333;
    }
    
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        border-top: 1px solid #e0e0e0;
        margin-top: 2rem;
    }
    
    .sidebar-info {
        background: #000000;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid #333333;
    }
</style>
""", unsafe_allow_html=True)

def load_model_info():
    """Load model info only (no actual model loading)"""
    model_info = {}
    if os.path.exists('model_info.json'):
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
    return model_info

def mock_prediction(input_data):
    """Mock prediction function for frontend demonstration"""
    # This is just for demo - real predictions would come from your backend API
    engagement_score = 0
    engagement_score += 1 if input_data['came_from_facebook'].lower() == 'yes' else 0
    engagement_score += 1 if input_data['follows_facebook_page'].lower() == 'yes' else 0
    engagement_score += 1 if input_data['heard_about_shop'].lower() == 'yes' else 0
    
    # Simple mock logic based on engagement and price
    price = input_data['sell_price']
    if engagement_score >= 2 and price > 20000:
        return 1, [0.3, 0.7]  # Returning customer with 70% probability
    elif engagement_score >= 1 and price > 15000:
        return 1, [0.45, 0.55]  # Borderline case
    else:
        return 0, [0.65, 0.35]  # New customer

def create_gauge_chart(probability):
    """Create a gauge chart for probability"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Return Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 50], 'color': "yellow"},
                {'range': [50, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "green"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(height=300, font={'color': "darkblue", 'family': "Arial"})
    return fig

def create_engagement_chart(came_from_fb, follows_page, heard_about):
    """Create engagement breakdown chart with smooth styling"""
    engagement_data = pd.DataFrame({
        'Factor': ['Facebook Visit', 'Page Follow', 'Brand Awareness'],
        'Active': [
            1 if came_from_fb == "Yes" else 0,
            1 if follows_page == "Yes" else 0,
            1 if heard_about == "Yes" else 0
        ]
    })
    
    colors = ['#FF6B6B' if x == 1 else '#E0E0E0' for x in engagement_data['Active']]
    
    fig = px.bar(
        engagement_data, 
        x='Factor', 
        y='Active',
        title="Social Media Engagement Status",
        color_discrete_sequence=colors
    )
    fig.update_layout(
        height=300, 
        showlegend=False, 
        yaxis={'range': [0, 1.2]},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(
        texttemplate='%{y}', 
        textposition='outside',
        marker_line_color='rgba(0,0,0,0.2)',
        marker_line_width=1
    )
    return fig

def create_feature_importance_chart():
    """Create a sample feature importance chart with smooth styling"""
    features = ['Age', 'Price', 'Engagement Score', 'Facebook Source', 'Page Follow', 'Brand Awareness']
    importance = [0.25, 0.30, 0.20, 0.10, 0.08, 0.07]
    
    fig = px.bar(
        x=importance,
        y=features,
        orientation='h',
        title="Feature Importance in Prediction",
        color=importance,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(
        marker_line_color='rgba(0,0,0,0.2)',
        marker_line_width=1
    )
    return fig

def create_customer_trend_chart():
    """Create a smooth line plot showing customer trends over time"""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    returning_customers = [65, 72, 68, 75, 78, 82, 85, 88, 91, 87, 89, 93]
    new_customers = [120, 135, 125, 140, 145, 150, 155, 160, 165, 158, 162, 170]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=returning_customers,
        mode='lines+markers',
        name='Returning Customers',
        line=dict(color='#11998e', width=4, shape='spline', smoothing=1.3),
        marker=dict(size=10, color='#11998e', line=dict(width=2, color='white')),
        fill='tonexty' if len(fig.data) > 0 else None,
        fillcolor='rgba(17, 153, 142, 0.1)'
    ))
    fig.add_trace(go.Scatter(
        x=months,
        y=new_customers,
        mode='lines+markers',
        name='New Customers',
        line=dict(color='#ee0979', width=4, shape='spline', smoothing=1.3),
        marker=dict(size=10, color='#ee0979', line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(238, 9, 121, 0.1)'
    ))
    
    fig.update_layout(
        title='Customer Acquisition & Retention Trends',
        xaxis_title='Month',
        yaxis_title='Number of Customers',
        height=400,
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def generate_customer_insights(age, gender, sell_price, engagement_score, probability):
    """Generate personalized customer insights"""
    insights = []
    
    if age < 25:
        insights.append("🧑‍💼 Young demographic - Focus on trendy features and social media marketing")
    elif age < 40:
        insights.append("👨‍💼 Working professional - Emphasize productivity and reliability features")
    else:
        insights.append("👴 Mature customer - Prioritize ease of use and customer support")
    
    if sell_price > 50000:
        insights.append("💎 Premium segment - Offer exclusive services and latest accessories")
    elif sell_price > 25000:
        insights.append("💰 Mid-premium range - Balance features with value proposition")
    else:
        insights.append("💵 Budget-conscious - Highlight cost savings and essential features")
    
    if engagement_score >= 2:
        insights.append("🌟 Highly engaged - Likely to respond to social media campaigns")
    elif engagement_score == 1:
        insights.append("📱 Moderately engaged - Opportunity to increase social interaction")
    else:
        insights.append("🎯 Low engagement - Needs targeted outreach and awareness building")
    
    if probability > 0.7:
        insights.append("✅ High retention potential - Invest in premium customer experience")
    elif probability > 0.4:
        insights.append("⚡ Moderate retention - Implement follow-up campaigns")
    else:
        insights.append("🚨 Low retention risk - Requires immediate attention and special offers")
    
    return insights

# Main App
def main():
    st.markdown('<h1 class="main-header">📱 Mobile Shop Customer Predictor</h1>', unsafe_allow_html=True)
    
    # Load model info only (no actual model)
    model_info = load_model_info()
    
    if model_info:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Model Accuracy", f"{model_info.get('accuracy', 0):.1%}")
        with col2:
            st.metric("📈 AUC Score", f"{model_info.get('auc_score', 0):.3f}")
        with col3:
            st.metric("🔧 Features Used", f"{model_info.get('n_features', 'N/A')}")
        with col4:
            training_date = model_info.get('training_date', '')
            if training_date:
                date_str = datetime.fromisoformat(training_date.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                st.metric("📅 Last Trained", date_str)
    
    st.markdown("### Predict if a customer will return based on their profile and engagement")
    
    # Sidebar for inputs
    st.sidebar.markdown("## 👤 Customer Information")
    st.sidebar.markdown('<div class="info-box">Fill in the customer details to get a prediction</div>', unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        age = st.number_input("👤 Age", min_value=16, max_value=80, value=25, step=1)
        gender = st.selectbox("⚥ Gender", ["Male", "Female"])
    with col2:
        sell_price = st.number_input("💰 Phone Price ($)", min_value=100, max_value=100000, value=20000, step=100)
        customer_location = st.selectbox("📍 Location", [
            "Dhaka", "Chittagong", "Sylhet", "Rajshahi", "Khulna", 
            "Barisal", "Rangpur", "Mymensingh", "Other"
        ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📱 Mobile Information")
    mobile_name = st.sidebar.selectbox("📱 Mobile Brand", [
        "Samsung", "iPhone", "Xiaomi", "Oppo", "Vivo", 
        "Realme", "OnePlus", "Huawei", "Nokia", "Other"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🌐 Social Media Engagement")
    came_from_facebook = st.sidebar.radio("📘 Came from Facebook?", ["Yes", "No"])
    follows_facebook_page = st.sidebar.radio("👍 Follows our Facebook page?", ["Yes", "No"])
    heard_about_shop = st.sidebar.radio("🔊 Heard about shop before?", ["Yes", "No"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="sidebar-info">
        💡 <strong>Pro Tip:</strong><br>
        Higher engagement scores typically lead to better customer retention rates!
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("🔮 Predict Customer Return", type="primary"):
        try:
            input_data = {
                'age': age,
                'gender': gender.lower(),
                'sell_price': sell_price,
                'came_from_facebook': came_from_facebook,
                'follows_facebook_page': follows_facebook_page,
                'heard_about_shop': heard_about_shop,
                'customer_location': customer_location.lower(),
                'mobile_name': mobile_name.lower()
            }
            
            # Calculate engagement metrics for display
            engagement_score = 0
            engagement_score += 1 if came_from_facebook.lower() == 'yes' else 0
            engagement_score += 1 if follows_facebook_page.lower() == 'yes' else 0
            engagement_score += 1 if heard_about_shop.lower() == 'yes' else 0
            
            engagement_level = "low" if engagement_score <= 1 else "medium" if engagement_score == 2 else "high"
            price_category = "low" if sell_price <= 15000 else "medium" if sell_price <= 30000 else "high"
            
            # Mock prediction (replace with API call to your backend in production)
            prediction, probability = mock_prediction(input_data)
            
            # Display results
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if prediction == 1:
                    st.markdown(f'''
                    <div class="prediction-result returning-customer">
                        🎉 RETURNING CUSTOMER!<br>
                        Probability: {probability[1]:.1%}
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="prediction-result new-customer">
                        🆕 NEW CUSTOMER<br>
                        Return Probability: {probability[1]:.1%}
                    </div>
                    ''', unsafe_allow_html=True)
            
            # Rest of your display code remains the same...
            # [Include all the visualization and insight generation code from the original]
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

    else:
        # Default view - Dashboard overview
        st.markdown("## 🚀 Welcome to the Customer Prediction Dashboard!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("📊 **Smart Predictions**\n\nCustomer return probability analysis")
        with col2:
            st.success("📈 **Business Insights**\n\nActionable recommendations")
        with col3:
            st.warning("🎯 **Data-Driven Marketing**\n\nOptimize campaigns")
        
        fig_trend = create_customer_trend_chart()
        st.plotly_chart(fig_trend, use_container_width=True)
        
        if model_info:
            st.markdown("---")
            st.markdown("## 📊 Model Performance Overview")
            col1, col2 = st.columns(2)
            with col1:
                if 'target_distribution' in model_info:
                    dist_data = pd.DataFrame({
                        'Customer Type': ['New Customers', 'Returning Customers'],
                        'Count': [
                            model_info['target_distribution'].get('new_customers', 0),
                            model_info['target_distribution'].get('returning_customers', 0)
                        ]
                    })
                    fig = px.pie(dist_data, values='Count', names='Customer Type', 
                               title='Training Data Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                metrics_data = pd.DataFrame({
                    'Metric': ['Accuracy', 'AUC Score'],
                    'Score': [model_info.get('accuracy', 0), model_info.get('auc_score', 0)]
                })
                fig = px.bar(metrics_data, x='Metric', y='Score', 
                           title='Model Performance Metrics')
                st.plotly_chart(fig, use_container_width=True)
        
        # How to use section
        st.markdown("---")
        st.markdown("## 📖 How to Use This Application")
        with st.expander("🔍 Step-by-Step Guide"):
            st.markdown("""
            1. **Enter Customer Information** 📝
               - Fill in age, gender, and location details
               - Specify the mobile phone price and brand
            
            2. **Social Media Engagement** 🌐
               - Indicate if customer came from Facebook
               - Check if they follow your Facebook page
               - Note if they've heard about your shop before
            
            3. **Generate Prediction** 🔮
               - Click the "Predict Customer Return" button
               - View the probability and classification result
            
            4. **Analyze Results** 📊
               - Review customer profile analysis
               - Check visual insights and charts
               - Read AI-generated recommendations
            
            5. **Export Data** 📥
               - Download results in CSV or JSON format
               - Use for further analysis or reporting
            """)
        
        with st.expander("💡 Tips for Best Results"):
            st.markdown("""
            - **Accurate Data Entry:** Ensure all information is correct
            - **Complete Social Media Info:** This significantly impacts predictions
            - **Regular Model Updates:** Retrain with new data periodically
            - **Follow Recommendations:** Implement suggested marketing actions
            - **Track Results:** Monitor actual customer behavior vs predictions
            """)

if __name__ == "__main__":
    main()

st.markdown("---")
st.markdown("""
<div class="footer">
    📱 <strong>Mobile Shop Customer Predictor</strong> | 
    Built with ❤️ using Streamlit<br>
    <small>© 2024 - Frontend demonstration only. Predictions are simulated.</small>
</div>
""", unsafe_allow_html=True)