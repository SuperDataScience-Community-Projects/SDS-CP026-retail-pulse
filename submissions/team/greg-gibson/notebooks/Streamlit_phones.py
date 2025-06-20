import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

model = joblib.load('phone_customer_again.pkl')
model_FB = joblib.load('phone_customer_FB.pkl')
FB_influence = 0  # Initialize FB_influence
scaler = joblib.load('scaler.pkl')

st.title('Welcome to the Phone Store! :phone:')

location = st.selectbox('Select customer location', ['Rangamati Sadar', 'Inside Rangamati', 'Outside Rangamati'])
gender = st.selectbox('Select gender', ['male', 'female'])
unscaled_age = st.slider('Select age', 10, 100)
selected_phone = st.selectbox('Select phone', ['Galaxy A55 5G 8/128', 'Galaxy M35 5G 8/128', 'Galaxy S24 Ultra 12/256', 'iPhone 16 Pro 256GB', 'iPhone 16 Pro Max 1TB', 'iQOO Neo 9 Pro 5G 12/256', 'iQOO Z7 5G 6/128', 'Moto G85 5G 8/128', 'Narzo N53 4/64', 'Note 11S 6/128', 'Note 14 Pro 5G 8/256', 'Pixel 7a 8/128', 'Pixel 8 Pro 12/256', 'R-70 Turbo 5G 6/128', 'Redmi Note 12 Pro 8/128', 'Vivo T3x 5G 8/128', 'Vivo Y200 5G 6/128'])
unscaled_sale = st.number_input('Enter sale price', min_value=10000, max_value=300000, value=12000, step=1000)
fb = st.checkbox('From Facebook page?')
follow = st.checkbox('Follow us on Facebook?')
# before = st.checkbox('Have you bought from us before?')
hear = st.checkbox('Did you hear of our shop before?')

# Encoding the inputs
is_male = 1 if gender == "male" else 0
is_local = 0 if location == 'Outside Rangamati' else 1
from_FB = int(fb)  # True → 1, False → 0
follows_page = int(follow)  
heard_of_shop = int(hear)
unscaled_age = int(unscaled_age)  # Ensure age is an integer

# Encoding the phone selection
phone_options = [
       'Mobile Name_Galaxy M35 5G 8/128',
       'Mobile Name_Galaxy S24 Ultra 12/256', 'Mobile Name_Moto G85 5G 8/128',
       'Mobile Name_Narzo N53 4/64', 'Mobile Name_Note 11S 6/128',
       'Mobile Name_Note 14 Pro 5G 8/256', 'Mobile Name_Pixel 7a 8/128',
       'Mobile Name_Pixel 8 Pro 12/256', 'Mobile Name_R-70 Turbo 5G 6/128',
       'Mobile Name_Redmi Note 12 Pro 8/128', 'Mobile Name_Vivo T3x 5G 8/128',
       'Mobile Name_Vivo Y200 5G 6/128', 'Mobile Name_iPhone 16 Pro 256GB',
       'Mobile Name_iPhone 16 Pro Max 1TB',
       'Mobile Name_iQOO Neo 9 Pro 5G 12/256', 'Mobile Name_iQOO Z7 5G 6/128'
]
phone_feature = f"Mobile Name_{selected_phone}"

# Create a phone brand feature vector
phone_vector = [1 if col == phone_feature else 0 for col in phone_options]

# Scale the numerical inputs
scaled_values = scaler.transform([[unscaled_age, unscaled_sale]])  # Returns 2 values: scaled age, scaled sales
Age, SellPrice = scaled_values[0]

# Final feature vector
features = np.array([[Age, SellPrice, from_FB, follows_page, heard_of_shop,
       is_local, is_male] + phone_vector])

# Button: Predict returning customer
if st.button("Predict Returning Customer"):
    probs = model.predict_proba(features)[0]
    prediction = model.predict(features)
    
    # Store results in session state
    st.session_state.returning_prob = probs[1]
    st.session_state.returning_pred = prediction[0]

# ✅ Display results if they exist
if "returning_pred" in st.session_state:
    st.write(f"Probability of returning customer: {st.session_state.returning_prob * 100:.2f}%")
    st.write("A returning customer? →", "Yes" if st.session_state.returning_pred == 1 else "No")
    
    # Use prediction as a feature for second model
    FB_influence = st.session_state.returning_pred
    features_FB = np.array([[Age, SellPrice, heard_of_shop,
                             is_local, is_male, FB_influence] + phone_vector])

    # Button: Predict Facebook influence
    if st.button("Predict Customer Influenced by FaceBook"):
        probs_FB = model_FB.predict_proba(features_FB)[0]
        prediction_FB = model_FB.predict(features_FB)
        st.write(f"Probability: {probs_FB[1]*100:.2f}%")
        st.write("Customer influenced by FB? →", "Yes" if prediction_FB[0] == 1 else "No")
