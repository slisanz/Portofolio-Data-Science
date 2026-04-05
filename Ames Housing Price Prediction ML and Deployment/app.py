import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_ames_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
    st.success("✅ Model and artifacts loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🏠 Ames Housing Price Prediction")
st.markdown("Predict house sale prices based on property features")

# Sidebar input
st.sidebar.header("Input Features")

# Compute feature importance from the trained model
importance = model.feature_importances_
imp_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=False)
top_features = imp_df.head(15)['feature'].tolist()

st.sidebar.markdown("### Most Important Features")
user_input = {}
for feat in top_features:
    user_input[feat] = st.sidebar.number_input(feat, value=0.0, step=1.0, format="%.0f")

# Fill remaining features with 0
for feat in feature_names:
    if feat not in user_input:
        user_input[feat] = 0.0

# Prediction button
if st.sidebar.button("Predict Price"):
    input_df = pd.DataFrame([user_input])[feature_names]
    prediction = model.predict(input_df)[0]
    
    # Custom large display (bigger than st.metric)
    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 20px 0;">
            <p style="font-size: 24px; color: #666;"> Predicted House Price</p>
            <p style="font-size: 64px; font-weight: bold; color: #2e7d32; margin: 0;">${prediction:,.0f}</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.balloons()
else:
    st.info("Please enter feature values in the sidebar, then click 'Predict Price'.")