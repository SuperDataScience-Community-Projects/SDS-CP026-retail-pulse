import os
from dotenv import load_dotenv
import streamlit as st
import requests
from streamlit_extras.let_it_rain import rain
from datetime import datetime

# load environment variables from .env file
load_dotenv()

# Get the BASE_URL from the environment variables
base_url = os.getenv("BASE_URL", "http://localhost:8001")

st.set_page_config(
    page_title="Retail Pulse Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Streamlit app title and description
st.title("🧠 SDS Community Project - Retail Pulse")
st.write("Fill the form below to predict.")

prediction_target = st.radio(
    "Which prediction would you like to make?",
    (
        "Returning Customer Prediction",
        "Facebook Page Customer Prediction",
        "Customer Cluster Prediction"
    )
)


def get_selection(label, options):
    return options[st.selectbox(label, list(options.keys()))]


gender_option = {"Female": 0, "Male": 1}
facebook_page_option = {"No": 0, "Yes": 1}
returning_customer_option = {"No": 0, "Yes": 1}
followed_page_option = {"No": 0, "Yes": 1}
heard_before_option = {"No": 0, "Yes": 1}
is_local_option = {"No": 0, "Yes": 1}

# Input fields
age = st.number_input("Age", min_value=10, max_value=100, value=30)
gender = get_selection("Gender", gender_option)  # 0 = Female, 1 = Male
sell_price = st.number_input("Sell Price", min_value=0.00, value=30000.00)

included_variable_1 = included_variable_2 = None
included_feature_1 = included_feature_2 = None

if prediction_target == "Returning Customer Prediction":
    endpoint = "api/predict/returning_customer"
    included_variable = "does_he_she_come_from_facebook_page"
    included_feature = get_selection("From Facebook Page?", facebook_page_option)
elif prediction_target == "Facebook Page Customer Prediction":
    endpoint = "api/predict/facebook_page_customer"
    included_variable = "did_he_she_buy_any_mobile_before"
    included_feature = get_selection("Returning Customer?", returning_customer_option)
else:
    endpoint = "api/predict/cluster"
    included_variable = None
    included_feature = None
    included_variable_1 = "does_he_she_come_from_facebook_page"
    included_feature_1 = get_selection("From Facebook Page?", facebook_page_option)
    included_variable_2 = "did_he_she_buy_any_mobile_before"
    included_feature_2 = get_selection("Returning Customer?", returning_customer_option)


followed_page = get_selection("Followed Our Page?", followed_page_option)
heard_before = get_selection("Heard of Shop Before?", heard_before_option)
is_local = get_selection("Is Local?", is_local_option)

# Mapping: user-friendly label → internal key
mobile_label_map = {
    "Samsung Galaxy M35 5G (8+128 GB)": "galaxy_m35_5g_8_128",
    "Samsung Galaxy S24 Ultra (12+256 GB)": "galaxy_s24_ultra_12_256",
    "Motorola G85 5G (8+128 GB)": "moto_g85_5g_8_128",
    "Narzo N53 (4+64 GB)": "narzo_n53_4_64",
    "Redmi Note 11S (6+128 GB)": "note_11s_6_128",
    "Note 14 Pro 5G (8+256 GB)": "note_14_pro_5g_8_256",
    "Google Pixel 7a (8+128 GB)": "pixel_7a_8_128",
    "Google Pixel 8 Pro (12+256 GB)": "pixel_8_pro_12_256",
    "Realme R70 Turbo 5G (6+128 GB)": "r_70_turbo_5g_6_128",
    "Redmi Note 12 Pro (8+128 GB)": "redmi_note_12_pro_8_128",
    "Vivo T3X 5G (8+128 GB)": "vivo_t3x_5g_8_128",
    "Vivo Y200 5G (6+128 GB)": "vivo_y200_5g_6_128",
    "iPhone 16 Pro (256 GB)": "iphone_16_pro_256gb",
    "iPhone 16 Pro Max (1TB)": "iphone_16_pro_max_1tb",
    "iQOO Neo 9 Pro 5G (12+256 GB)": "iqoo_neo_9_pro_5g_12_256",
    "iQOO Z7 5G (6+128 GB)": "iqoo_z7_5g_6_128"
}
# Select mobile by label
selected_label = st.selectbox("Select Mobile", list(mobile_label_map.keys()))
selected_key = mobile_label_map[selected_label]
# Initialize all to False
mobiles = {value: False for value in mobile_label_map.values()}
mobiles[selected_key] = True

# 📅 User selects a date
selected_date = st.date_input("Select a Date", datetime.today())

# 🔍 Extract values in backend
day_of_week = selected_date.weekday()  # 0 = Monday, 6 = Sunday
month = selected_date.month            # 1 = January, 12 = December
is_weekend = 1 if day_of_week >= 5 else 0  # 1 = Weekend (Saturday/Sunday), 0 = Weekday

# Prediction button
if st.button("🔥🚀🔥 Predict 🔥🚀🔥"):
    # Prepare the data for API request
    payload = {
        "age": age,
        "gender": gender,
        "sell_price": sell_price,
        "does_he_she_followed_our_page": followed_page,
        "did_he_she_hear_of_our_shop_before": heard_before,
        "is_local": is_local,
        **{f"mobile_name_{k}": v for k, v in mobiles.items()},
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend
    }

    # Add included_variable only for classification endpoints
    if included_variable and included_feature is not None:
        payload[included_variable] = included_feature
    else:
        payload[included_variable_1] = included_feature_1
        payload[included_variable_2] = included_feature_2

    # Make the API request
    response = requests.post(f"{base_url}/{endpoint}", json=payload)
    if response.status_code == 200:
        prediction = response.json()
        st.success(f"The prediction is that the client is {prediction['label']} "
                   f"with (Class: {prediction['prediction']})",
                   icon=":material/thumb_up:")
        rain(emoji="🎈", font_size=54, falling_speed=5, animation_length="infinite", )
    else:
        st.write("Error: Could not retrieve prediction. Please try again.")

# Run the Streamlit App
# streamlit run app.py
