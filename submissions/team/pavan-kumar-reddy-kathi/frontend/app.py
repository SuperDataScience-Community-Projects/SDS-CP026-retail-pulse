import datetime
import os
from dotenv import load_dotenv
import streamlit as st
import requests
from streamlit_extras.let_it_rain import rain
import logging

# load env variables
load_dotenv()

st.title('Retail Pulse - Customer Journey Prediction & Marketing Impact Analysis')
st.header('Complete the form below and click on Predict button')

prediction_options = ["Customer Clustering", "Is Returning Customer",
                                            "Is Purchase Due to Facebook Marketing"]
prediction = st.radio("Prediction To Perform",prediction_options , index=0)

today = datetime.date.today()
purchase_date = st.date_input('Date of Purchase', value=today, max_value=today)

location = st.radio("Location", ["Inside Rangamati", "Rangamati Sadar",
                                            "Outside Rangamati"], index=0)

age = st.number_input('Age', min_value=1, max_value=130, step=1)

gender = st.radio('Gender', ['Female', 'Male'], index=0)

mobile_model = st.selectbox('Mobile Model', ["Galaxy A55 5G 8/128", "Galaxy S24 Ultra 12/256", "Galaxy M35 5G 8/128",
                                            "iPhone 16 Pro 256GB", "iPhone 16 Pro Max 1TB", "iQOO Neo 9 Pro 5G 12/256",
                                            "iQOO Z7 5G 6/128", "Moto G85 5G 8/128", "Narzo N53 4/64", "Note 11S 6/128",
                                            "Note 14 Pro 5G 8/256", "Pixel 7a 8/128", "Pixel 8 Pro 12/256",
                                            "Redmi Note 12 Pro 8/128", "R-70 Turbo 5G 6/128", "Vivo T3x 5G 8/128",
                                            "Vivo Y200 5G 6/128"])

price = st.number_input('Mobile Price',min_value=1)

if prediction in ["Customer Clustering", "Is Returning Customer"]:
    is_from_facebook_page = st.radio('Does the customer come from the Facebook page?', ['Yes', 'No'], index=1)

if prediction in ["Customer Clustering", "Is Purchase Due to Facebook Marketing"]:
    is_returning_customer = st.radio('Did the customer buy any mobile before?', ['Yes', 'No'], index=1)

is_facebook_page_follower = st.radio('Does the customer Followed Our Page?', ['Yes', 'No'], index=1)
is_customer_already_aware_about_shop = st.radio('Did the customer hear of our shop before?', ['Yes', 'No'], index=1)

# Predict button click
if st.button("Predict"):
    ui_selections = {
        'purchase_date' :  purchase_date.isoformat(),
        'location' : location,
        'age' : age,
        'gender' : gender[0],
        'mobile_name' : mobile_model,
        'sale_price' : price,
        'is_facebook_page_follower' : is_facebook_page_follower == 'Yes',
        'customer_already_know_about_shop':is_customer_already_aware_about_shop == 'Yes'
    }
    endpoint = ''
    if prediction == "Customer Clustering":
        ui_selections['is_from_facebook_page'] = is_from_facebook_page == 'Yes'
        ui_selections['is_returning_customer'] = is_returning_customer == 'Yes'
        endpoint = 'predict-customer-cluster'

    elif prediction == "Is Returning Customer":
        ui_selections['is_from_facebook_page'] = is_from_facebook_page == 'Yes'
        endpoint = 'predict-returning-customer'
    else:
        ui_selections['is_returning_customer'] = is_returning_customer == 'Yes'
        endpoint = 'predict-facebook-marketing-impact'

    api_host = os.getenv('RETAIL_PULSE_PREDICTION_API_BASE_URL')
    api_endpoint = f'{api_host}/api/{endpoint}'
    response = requests.post(api_endpoint, json=ui_selections)

    prediction_text = ''
    if response is not None and response.status_code == 200:
        if prediction == "Customer Clustering":
            prediction_text = response.text
        else:
            prediction_text = 'Yes' if response.text == "1" else 'No'

        st.success(f'**{prediction} : {prediction_text}**', icon='✔')
        # Include emoji using Windows Key + . (period)
        rain(emoji='📱', font_size=50, falling_speed=5, animation_length='infinite')
    else:
        logging.error(f'Error response {response.json() if response is not None else None} for request {ui_selections}')
        st.error(f'No Response or Invalid HTTP Status Code. Please retry with valid values.')










