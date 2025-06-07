import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import streamlit as st
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
    initial_sidebar_state="expanded"
)

st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio("Go to", ["📌 Make a Prediction", "📊 View Dashboards"])


# ==========================================================
# Load or Simulate Data for Dashboard
# ==========================================================
@st.cache_data
def load_data():
    url = base_url + "/api/dataset"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        dataset = pd.DataFrame(data)
        return dataset
    else:
        st.error("Failed to load dataset from API")
        return pd.DataFrame()


# ==========================================================
# Dashboard Section
# ==========================================================
if menu == "📊 View Dashboards":
    st.title("📊 Retail Pulse Dashboards")

    df = load_data()

    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Reverse one-hot encoded mobile name columns into a single column
    def reverse_one_hot(row):
        for col in row.index:
            if col.startswith("mobile_name_") and row[col] == True:
                return col.replace("mobile_name_", "").replace("_", " ").upper()
        return "GALAXY A55 5G 8 128"


    df["mobile_name"] = df.apply(reverse_one_hot, axis=1)

    # Convert gender to readable format
    df["gender"] = df["gender"].map({0: "Female", 1: "Male"})

    # Convert is_local to readable format for display (optional)
    df["location_type"] = df["is_local"].map({1: "Local", 0: "Non-local"})

    # --- Sidebar Filters ---
    st.sidebar.header("🔍 Filter Options")

    # Date range
    min_date, max_date = df["date"].min(), df["date"].max()
    start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date,
                                                 max_value=max_date)

    # Gender filter
    genders = st.sidebar.multiselect("Select Gender", options=df["gender"].unique(), default=df["gender"].unique())

    # Local / Non-local
    locality = st.sidebar.multiselect("Select Location Type", options=df["location_type"].unique(),
                                      default=df["location_type"].unique())

    # Advanced filters
    st.sidebar.markdown("### 🧠 Advanced Filters")
    fb_source = st.sidebar.selectbox("From Facebook Page?", options=["All", "Yes", "No"])
    follows_page = st.sidebar.selectbox("Follows Our Page?", options=["All", "Yes", "No"])
    bought_before = st.sidebar.selectbox("Bought Before?", options=["All", "Yes", "No"])
    heard_before = st.sidebar.selectbox("Heard of Shop Before?", options=["All", "Yes", "No"])

    # Apply filters
    df_filtered = df[
        (df["date"] >= pd.to_datetime(start_date)) &
        (df["date"] <= pd.to_datetime(end_date)) &
        (df["gender"].isin(genders)) &
        (df["location_type"].isin(locality))
        ]

    # Advanced filter logic
    if fb_source != "All":
        df_filtered = df_filtered[
            df_filtered["does_he_she_come_from_facebook_page"] == (1 if fb_source == "Yes" else 0)]

    if follows_page != "All":
        df_filtered = df_filtered[df_filtered["does_he_she_followed_our_page"] == (1 if follows_page == "Yes" else 0)]

    if bought_before != "All":
        df_filtered = df_filtered[
            df_filtered["did_he_she_buy_any_mobile_before"] == (1 if bought_before == "Yes" else 0)]

    if heard_before != "All":
        df_filtered = df_filtered[
            df_filtered["did_he_she_hear_of_our_shop_before"] == (1 if heard_before == "Yes" else 0)]

    # --- Dashboard ---

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Sales", f"${df_filtered['sell_price'].sum():,.2f}")

    with col2:
        st.metric("Total Customers", df_filtered.shape[0])

    with col3:
        st.metric("Average Price", f"${df_filtered['sell_price'].mean():,.2f}")

    # Tabs for organized display
    tab1, tab2, tab3 = st.tabs(["📈 Trends", "📦 Sales Insights", "👥 Demographics"])

    with tab1:
        st.subheader("📈 Sales Trend Over Time")
        df_time = df_filtered.groupby("date")["sell_price"].sum().reset_index()
        fig_time = px.line(df_time, x="date", y="sell_price", title="Sales Over Time", markers=True)
        st.plotly_chart(fig_time, use_container_width=True)

        st.subheader("💰 Price Distribution by Gender")
        fig_box = px.box(
            df_filtered,
            x="gender",
            y="sell_price",
            color="gender",
            title="Sell Price by Gender")
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("💵 Total Sales by Gender")
        gender_sales = df_filtered.groupby("gender")["sell_price"].sum().reset_index()
        gender_sales["gender"] = gender_sales["gender"].map({0: "Female", 1: "Male"})

        fig_total = px.bar(
            gender_sales,
            x="gender",
            y="sell_price",
            color="gender",
            labels={"sell_price": "Total Sell Price", "gender": "Gender"},
            title="Total Sell Price by Gender",
            text_auto=".2s"
        )
        st.plotly_chart(fig_total, use_container_width=True)
        st.write(gender_sales)
        st.write(gender_sales["gender"])
    with tab2:
        st.subheader("📦 Units Sold by Mobile Model")
        df_mobile_popularity = df_filtered["mobile_name"].value_counts().reset_index()
        df_mobile_popularity.columns = ["mobile_name", "count"]
        fig_popularity = px.bar(
            df_mobile_popularity,
            x="mobile_name", y="count",
            labels={"index": "Mobile Model", "mobile_name": "Units Sold"},
            title="Top Selling Mobile Models"
        )
        st.plotly_chart(fig_popularity, use_container_width=True)

        st.subheader("📊 Spend Based on Purchase History")
        fig_prev = px.box(
            df_filtered,
            x="did_he_she_buy_any_mobile_before",
            y="sell_price",
            title="Sell Price by Previous Purchase",
            labels={"did_he_she_buy_any_mobile_before": "Bought Before (0 = No, 1 = Yes)"}
        )
        st.plotly_chart(fig_prev, use_container_width=True)

        # Sales by mobile model
        st.subheader("Sales by Mobile Model")
        fig_model = px.bar(
            df_filtered.groupby("mobile_name")["sell_price"].sum().sort_values(ascending=True).reset_index(),
            x="sell_price", y="mobile_name", orientation="h",
            labels={"sell_price": "Total Sales", "mobile_name": "Mobile Model"},
        )
        st.plotly_chart(fig_model, use_container_width=True)

    with tab3:
        st.subheader("👥 Customer Age Distribution")
        fig_age = px.histogram(df_filtered, x="age", nbins=10, title="Distribution of Customer Ages")
        st.plotly_chart(fig_age, use_container_width=True)

        # Sales by gender
        st.subheader("🧑‍🤝‍🧑 Sales by Gender")
        fig_gender = px.pie(df_filtered, names="gender", values="sell_price", title="Sales by Gender")
        st.plotly_chart(fig_gender, use_container_width=True)

        # Sales by Local vs Non-local
        st.subheader("Sales by Location Type")
        fig_local = px.pie(df_filtered, names="location_type", values="sell_price",
                           title="Sales by Location (Local vs Non-local)")
        st.plotly_chart(fig_local, use_container_width=True)

    # --- Data Preview & Download ---
    st.subheader("📄 Filtered Dataset Preview")
    st.dataframe(df_filtered)

    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )


# =============================
# Prediction Section
# =============================

elif menu == "📌 Make a Prediction":
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
