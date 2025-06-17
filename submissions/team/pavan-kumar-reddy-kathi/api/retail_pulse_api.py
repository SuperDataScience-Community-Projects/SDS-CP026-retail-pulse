import calendar
from pydantic import BaseModel
from datetime import date
from enums import Location, Gender, MobileName
import joblib
import numpy as np
from fastapi import FastAPI
import os

# Instantiate FastAPI
app = FastAPI(title='Retail Pulse API',
              description='Contains API methods to Cluster, Predict retuning customers and impact '
                          'of Facebook marketing', version='1.0.0')

# Load models
facebook_marketing_model = joblib.load(os.path.join(os.path.dirname(__file__),'models','facebook_marketing.pkl'))
returning_customer_model = joblib.load(os.path.join(os.path.dirname(__file__),'models','returning_customer.pkl'))
customer_cluster_model = joblib.load(os.path.join(os.path.dirname(__file__), 'models', 'customer_clusters.pkl'))

# Define cluster names
cluster_names = {
    0: "Practical Professionals",
    1: "Ambitious Tech Enthusiasts",
    2: "Budget-Conscious Youngsters",
    3: "Elite Tech Aficionados"
}

class MobilePurchaseInfo(BaseModel):
    """
    Base Model that captures purchase information of a mobile
    """
    purchase_date:date
    location:Location
    age:int
    gender:Gender
    mobile_name:MobileName
    sale_price:float
    is_facebook_page_follower:bool
    customer_already_know_about_shop:bool

class FaceBookMarketingModel(MobilePurchaseInfo):
    """
    Input payload used in predicting impact of facebook marketing on mobile purchase
    """
    is_returning_customer:bool

class ReturningCustomerModel(MobilePurchaseInfo):
    """
    Input payload used in predicting whether buyer is returning customer or not
    """
    is_from_facebook_page:bool

class ClusteringModel(MobilePurchaseInfo):
    """
    Input payload used in predicting cluster for buyer
    """
    is_from_facebook_page: bool
    is_returning_customer: bool


def get_sine_cosine_transformations(input_date:date):
    """
    Extracts day of year from input date and applies sin and cos transformations,
    and returns the corresponding values as output
    :param input_date: Input date object
    :return: sin and cos values obtained after applying sin and cos transformations on day of year
    extracted from input date
    """
    day_of_year = input_date.timetuple().tm_yday
    days_in_year = 365 + calendar.isleap(input_date.year)
    sin_day_of_year = np.sin(2 * np.pi * day_of_year / days_in_year)
    cos_day_of_year = np.cos(2 * np.pi * day_of_year / days_in_year)
    return sin_day_of_year, cos_day_of_year


def is_local_customer(location:Location)->int:
    """
    Extract Local Vs Non-Local information from input Location enum
    :param location: Instance of Location enum
    :return: 0 if location is Outside Rangamati else 1
    """
    return 0 if location == Location.Outside_Rangamati else 1

def is_female_customer(gender:Gender)->int:
    """
    Extract male Vs female information from input Gender enum
    :param gender: Instance of Gender enum
    :return: 1 if gender is female else 0
    """
    return 1 if gender == Gender.Female else 0

@app.post(
    path='/api/predict-facebook-marketing-impact',
    summary='Predicts impact of facebook marketing',
    description='Predicts impact of facebook marketing based on input mobile purchase information',
    tags=["Prediction"]
)
def predict_facebook_marketing_impact(request:FaceBookMarketingModel)->int:
    """
    API endpoint for predicting the impact of facebook marketing on
    input mobile purchase transaction
    :param request: Instance of FaceBookMarketingModel
    :return: 1 , if model predicts that input mobile purchase is originated from facebook
    else 0
    """
    is_local = is_local_customer(request.location)
    sin_day_of_year, cos_day_of_year = get_sine_cosine_transformations(request.purchase_date)
    is_female = is_female_customer(request.gender)
    matrix_feature = np.array([[
        request.age,
        request.sale_price,
        request.mobile_name,
        is_local,
        is_female,
        1 if request.is_facebook_page_follower else 0,
        1 if request.is_returning_customer else 0,
        1 if request.customer_already_know_about_shop else 0,
        sin_day_of_year,
        cos_day_of_year
    ]])
    y_predict = facebook_marketing_model.predict(matrix_feature)
    return int(y_predict[0])

@app.post(
    path='/api/predict-returning-customer',
    summary='Predicts whether customer is retuning customer or not',
    description='Predicts whether customer is retuning customer or not based on input mobile purchase information',
    tags=["Prediction"]
)
def predict_returning_customer(request:ReturningCustomerModel)->int:
    """
    API endpoint for predicting whether the input mobile purchase transaction is done
    by returning customer or not
    :param request: Instance of ReturningCustomerModel
    :return: 1 , if model predicts that input mobile purchase is done by returning customer
    else 0
    """
    is_local = is_local_customer(request.location)
    sin_day_of_year, cos_day_of_year = get_sine_cosine_transformations(request.purchase_date)
    is_female = is_female_customer(request.gender)
    matrix_feature = np.array([[
        request.age,
        request.sale_price,
        request.mobile_name,
        is_local,
        is_female,
        1 if request.is_from_facebook_page else 0,
        1 if request.is_facebook_page_follower else 0,
        1 if request.customer_already_know_about_shop else 0,
        sin_day_of_year,
        cos_day_of_year
    ]], dtype=object)
    y_predict = returning_customer_model.predict(matrix_feature)
    return int(y_predict[0])

@app.post(
    path='/api/predict-customer-cluster',
    summary='Predicts the cluster to which input customer belongs to',
    description='Predicts the cluster to which input customer belongs to',
    tags=["Prediction"]
)
def predict_customer_cluster(request:ClusteringModel)->str:
    """
    API endpoint that Predicts the cluster to which input customer belongs to
    :param request: Instance of ClusteringModel
    :return: string representing cluster name to which input customer belongs to
    """
    is_local = is_local_customer(request.location)
    sin_day_of_year, cos_day_of_year = get_sine_cosine_transformations(request.purchase_date)
    is_female = is_female_customer(request.gender)
    matrix_feature = np.array([[
        request.age,
        request.sale_price,
        request.mobile_name,
        is_local,
        is_female,
        1 if request.is_from_facebook_page else 0,
        1 if request.is_facebook_page_follower else 0,
        1 if request.customer_already_know_about_shop else 0,
        sin_day_of_year,
        cos_day_of_year,
        1 if request.is_returning_customer else 0
    ]])
    y_predict = customer_cluster_model.predict(matrix_feature)
    return cluster_names[int(y_predict[0])]


