from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn


# Load the trained models and scalers
model_returning_customer = joblib.load("models/returning_customer.pkl")
model_facebook_customer = joblib.load("models/facebook_customer.pkl")
model_cluster = joblib.load("models/kmeans.pkl")
scaler_returning_customer = joblib.load("models/returning_customer_scaler.pkl")
scaler_facebook_customer = joblib.load("models/facebook_customer_scaler.pkl")
scaler_cluster = joblib.load("models/kmeans_scaler.pkl")

# define FastAPI app
app = FastAPI(
    title="SDS Community Project 26 Retail Pulse API",
    description="An API for predicting real-world mobile sales data collected by TechCorner, "
                "a retail store in Rangamati, Bangladesh.",
    version="1.0.0"
)


# Define input schema
class ReturningCustomerInputData(BaseModel):
    age: int
    gender: int
    sell_price: float
    does_he_she_come_from_facebook_page: int
    does_he_she_followed_our_page: int
    did_he_she_hear_of_our_shop_before: int
    is_local: int
    mobile_name_galaxy_m35_5g_8_128: bool
    mobile_name_galaxy_s24_ultra_12_256: bool
    mobile_name_moto_g85_5g_8_128: bool
    mobile_name_narzo_n53_4_64: bool
    mobile_name_note_11s_6_128: bool
    mobile_name_note_14_pro_5g_8_256: bool
    mobile_name_pixel_7a_8_128: bool
    mobile_name_pixel_8_pro_12_256: bool
    mobile_name_r_70_turbo_5g_6_128: bool
    mobile_name_redmi_note_12_pro_8_128: bool
    mobile_name_vivo_t3x_5g_8_128: bool
    mobile_name_vivo_y200_5g_6_128: bool
    mobile_name_iphone_16_pro_256gb: bool
    mobile_name_iphone_16_pro_max_1tb: bool
    mobile_name_iqoo_neo_9_pro_5g_12_256: bool
    mobile_name_iqoo_z7_5g_6_128: bool
    day_of_week: int
    month: int
    is_weekend: int


class FacebookCustomerInputData(BaseModel):
    age: int
    gender: int
    sell_price: float
    does_he_she_followed_our_page: int
    did_he_she_buy_any_mobile_before: int
    did_he_she_hear_of_our_shop_before: int
    is_local: int
    mobile_name_galaxy_m35_5g_8_128: bool
    mobile_name_galaxy_s24_ultra_12_256: bool
    mobile_name_moto_g85_5g_8_128: bool
    mobile_name_narzo_n53_4_64: bool
    mobile_name_note_11s_6_128: bool
    mobile_name_note_14_pro_5g_8_256: bool
    mobile_name_pixel_7a_8_128: bool
    mobile_name_pixel_8_pro_12_256: bool
    mobile_name_r_70_turbo_5g_6_128: bool
    mobile_name_redmi_note_12_pro_8_128: bool
    mobile_name_vivo_t3x_5g_8_128: bool
    mobile_name_vivo_y200_5g_6_128: bool
    mobile_name_iphone_16_pro_256gb: bool
    mobile_name_iphone_16_pro_max_1tb: bool
    mobile_name_iqoo_neo_9_pro_5g_12_256: bool
    mobile_name_iqoo_z7_5g_6_128: bool
    day_of_week: int
    month: int
    is_weekend: int


class ClusterInputData(BaseModel):
    age: int
    gender: int
    sell_price: float
    does_he_she_come_from_facebook_page: int
    does_he_she_followed_our_page: int
    did_he_she_buy_any_mobile_before: int
    did_he_she_hear_of_our_shop_before: int
    is_local: int
    mobile_name_galaxy_m35_5g_8_128: bool
    mobile_name_galaxy_s24_ultra_12_256: bool
    mobile_name_moto_g85_5g_8_128: bool
    mobile_name_narzo_n53_4_64: bool
    mobile_name_note_11s_6_128: bool
    mobile_name_note_14_pro_5g_8_256: bool
    mobile_name_pixel_7a_8_128: bool
    mobile_name_pixel_8_pro_12_256: bool
    mobile_name_r_70_turbo_5g_6_128: bool
    mobile_name_redmi_note_12_pro_8_128: bool
    mobile_name_vivo_t3x_5g_8_128: bool
    mobile_name_vivo_y200_5g_6_128: bool
    mobile_name_iphone_16_pro_256gb: bool
    mobile_name_iphone_16_pro_max_1tb: bool
    mobile_name_iqoo_neo_9_pro_5g_12_256: bool
    mobile_name_iqoo_z7_5g_6_128: bool
    day_of_week: int
    month: int
    is_weekend: int


# Define prediction endpoint
@app.post(
    "/api/predict/returning_customer",
    summary="Predict Returning Customer",
    description="Predict whether a customer would return or not.",
    tags=["Prediction"]
)
def predict_returning_customer(data: ReturningCustomerInputData):
    """
    :param data: ReturningCustomerInputData:
    - age
    - gender
    - sell_price
    - does_he_she_come_from_facebook_page
    - does_he_she_followed_our_page
    - did_he_she_hear_of_our_shop_before
    - is_local
    - mobile_name_galaxy_m35_5g_8_128
    - mobile_name_galaxy_s24_ultra_12_256
    - mobile_name_moto_g85_5g_8_128
    - mobile_name_narzo_n53_4_64
    - mobile_name_note_11s_6_128
    - mobile_name_note_14_pro_5g_8_256
    - mobile_name_pixel_7a_8_128
    - mobile_name_pixel_8_pro_12_256
    - mobile_name_r_70_turbo_5g_6_128
    - mobile_name_redmi_note_12_pro_8_128
    - mobile_name_vivo_t3x_5g_8_128
    - mobile_name_vivo_y200_5g_6_128
    - mobile_name_iphone_16_pro_256gb
    - mobile_name_iphone_16_pro_max_1tb
    - mobile_name_iqoo_neo_9_pro_5g_12_256
    - mobile_name_iqoo_z7_5g_6_128
    - day_of_week
    - month
    - is_weekend

    :return:
    - Prediction: 0 for NO, 1 for YES.
    - Label: String representation of the prediction (NOT Returning Customer/Returning Customer).
    """
    input_features = pd.DataFrame([data.model_dump()])

    # Scale the input data
    scaled_data = scaler_returning_customer.transform(input_features)

    # Make the prediction
    prediction = model_returning_customer.predict(scaled_data)[0]
    label = "Not Returning Customer" if prediction == 0 else "Returning Customer"

    return {
        "prediction": int(prediction),
        "label": label
    }


@app.post(
    "/api/predict/facebook_page_customer",
    summary="Predict Facebook Page Customer",
    description="Predict whether a customer would come from our facebook page or not.",
    tags=["Prediction"]
)
def predict_facebook_customer(data: FacebookCustomerInputData):
    """
    :param data: FacebookCustomerInputData:
    - age
    - gender
    - sell_price
    - does_he_she_followed_our_page
    - did_he_she_buy_any_mobile_before
    - did_he_she_hear_of_our_shop_before
    - is_local
    - mobile_name_galaxy_m35_5g_8_128
    - mobile_name_galaxy_s24_ultra_12_256
    - mobile_name_moto_g85_5g_8_128
    - mobile_name_narzo_n53_4_64
    - mobile_name_note_11s_6_128
    - mobile_name_note_14_pro_5g_8_256
    - mobile_name_pixel_7a_8_128
    - mobile_name_pixel_8_pro_12_256
    - mobile_name_r_70_turbo_5g_6_128
    - mobile_name_redmi_note_12_pro_8_128
    - mobile_name_vivo_t3x_5g_8_128
    - mobile_name_vivo_y200_5g_6_128
    - mobile_name_iphone_16_pro_256gb
    - mobile_name_iphone_16_pro_max_1tb
    - mobile_name_iqoo_neo_9_pro_5g_12_256
    - mobile_name_iqoo_z7_5g_6_128
    - day_of_week
    - month
    - is_weekend

    :return:
    - Prediction: 0 for NO, 1 for YES.
    - Label: String representation of the prediction (NOT Returning Facebook Customer/Returning Facebook Customer).
    """
    # prepare the data for prediction
    input_features = pd.DataFrame([data.model_dump()])

    # Scale the input data
    scaled_data = scaler_facebook_customer.transform(input_features)

    # Make the prediction
    prediction = model_facebook_customer.predict(scaled_data)[0]
    label = "Not Returning Facebook Customer" if prediction == 0 else "Returning Facebook Customer"

    return {
        "prediction": int(prediction),
        "label": label
    }


@app.post(
    "/api/predict/cluster",
    summary="Predict Customer's Cluster",
    description="Predict cluster or segment a customer belongs to.",
    tags=["Prediction"]
)
def predict_customer_cluster(data: ClusterInputData):
    """
    :param data: ClusterInputData:
    - age
    - gender
    - sell_price
    - does_he_she_come_from_facebook_page
    - does_he_she_followed_our_page
    - did_he_she_buy_any_mobile_before
    - did_he_she_hear_of_our_shop_before
    - is_local
    - mobile_name_galaxy_m35_5g_8_128
    - mobile_name_galaxy_s24_ultra_12_256
    - mobile_name_moto_g85_5g_8_128
    - mobile_name_narzo_n53_4_64
    - mobile_name_note_11s_6_128
    - mobile_name_note_14_pro_5g_8_256
    - mobile_name_pixel_7a_8_128
    - mobile_name_pixel_8_pro_12_256
    - mobile_name_r_70_turbo_5g_6_128
    - mobile_name_redmi_note_12_pro_8_128
    - mobile_name_vivo_t3x_5g_8_128
    - mobile_name_vivo_y200_5g_6_128
    - mobile_name_iphone_16_pro_256gb
    - mobile_name_iphone_16_pro_max_1tb
    - mobile_name_iqoo_neo_9_pro_5g_12_256
    - mobile_name_iqoo_z7_5g_6_128
    - day_of_week
    - month
    - is_weekend

    :return:
    - Prediction: 0: "Mid-Tier/Moderate", 1: "Premium Shoppers",2: "Budget/Varied Shoppers" .
    - Label: String representation of the prediction (Mid-Tier/Moderate, Premium Shoppers, Budget/Varied Shoppers).
    """
    cluster_labels = {
        0: "Mid-Tier/Moderate",
        1: "Premium Shoppers",
        2: "Budget/Varied Shoppers"
    }

    # prepare the data for prediction
    input_features = pd.DataFrame([data.model_dump()])

    # Scale the input data
    scaled_data = scaler_cluster.transform(input_features)

    # Make the prediction
    prediction = int(model_cluster.predict(scaled_data)[0])
    label = cluster_labels.get(prediction, "Unknown")

    return {
        "prediction": int(prediction),
        "label": label
    }


@app.get(
    "/api/dataset",
    summary="Clean Dataset",
    description="Clean Dataset after EDA.",
    tags=["Dataset"]
)
def get_cleaned_data():
    try:
        df = pd.read_csv("data/cleaned_dataset.csv")  # Ensure correct path
        return df.to_dict(orient="records")  # Convert DataFrame to JSON
    except FileNotFoundError:
        return {"error": "File not found"}


# Run with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
