# Retail Pulse - Customer Journey Prediction & Marketing Impact Analysis

## Description
Retail Pulse is a machine learning project designed to extract actionable insights from 10 months of real-world mobile sales data collected by TechCorner, a retail store in Rangamati, Bangladesh. The primary focus is to understand and predict customer journeys by

  * Distinguishing between new and returning buyers
  * Evaluating the effectiveness of Facebook marketing
  * Segmenting customers based on behavioral patterns

By leveraging classification, clustering, and visual analytics, this project aims to deliver a comprehensive view of customer engagement, digital marketing performance, and purchasing trends.

Link to Dataset: https://www.kaggle.com/datasets/shohinurpervezshohan/techcorner-mobile-purchase-and-engagement-data

## Project Structure

| Folder/File Name   | Description                                                                |
|---------------|----------------------------------------------------------------------------|
| api           | Contains FASTAPI code that host final machine learning models as rest api  |
| data_analysis | Contains ipynb files with EDA and cluster analysis            |
| frontend      | Streamlit application the enables end users to try out these models via Fast API        |
| auto-machine-learning.py | Contains experimenation code with AutoML for Facebook marketing model                |
| dataset_module.py          | Contains code to load dataset from csv file, along with few reusable transformations        |
| facebook_marketing_model.py    | Contains Final prediction model (CatBoost) implementation for facebook marketing       |
| kmeans_clustering.py      | Contains KMeans Model implementation with four clusters          |
| optuna_hyper_parameter_tuning.ipynb        | Contains hyper parameter tuning code for various machine learning models using optuna  |
| returning_customer_model.py    | Contains Final prediction model (XgBoost) implementation for returning customers       |
| TechCorner_Sales_update.csv    | Dataset containing sales information for mobiles       |

## Installation
* If you are looking to install and run, all applications and modules then perform below steps. 
  * Open command prompt and change the current working directory to SDS-CP026-retail-pulse\submissions\team\pavan-kumar-reddy-kathi
  * Once you are in pavan-kumar-reddy-kathi directory, run **pip install -r requirements.txt**
* Streamlit application requires Fastapi to be up and running inorder to be functional.
* For FastAPI deployment, change current working directory to **api** folder & then run **uvicorn retail_pulse_api:app --reload** command
* Update .env entry (**RETAIL_PULSE_PREDICTION_API_BASE_URL**) in Streamlit app to point to FastAPI url (http://127.0.0.1:8000)
* For Streamlit application deployment, change current working directory to **frontend** folder & run **streamlit run app.py** command.
* You can also run FASTAPI as stand alone application, by only installing required dependencies from requirements.txt file under api folder 

## Dataset
Source to Dataset: https://www.kaggle.com/datasets/shohinurpervezshohan/techcorner-mobile-purchase-and-engagement-data

| Column Name   | Description                                                                                             |
|---------------|---------------------------------------------------------------------------------------------------------|
| Cus.ID           | Customer Id                  |
| Date         | Mobile Purchase date                               |
| Cus. Location       | Customer Location |
| Age         | Customer Age                                                                     |
| Gender         | Customer Gender                                                               |
| Mobile Name         | Mobile Name/Model                |
| Sell Price         | Purchase Price of Mobile                                                                              |
| Does he/she Come from Facebook Page?   | Target variable that indicates whether purchase is originated from Facebook |
| Does he/she Followed Our Page? | Indicates whether customer is follower of Mobile shop Facebook page |
| Did he/she buy any mobile before?      | Target Variable that indicates whether customer is returning customer or brand new customer |
| Did he/she hear of our shop before?      | Indicates whether customer is already aware about mobile shop |

## Exploratory Data Analysis
* Though there is no missing(na/null/nan) data present in dataset, it is observed that there are few rows (around 20) with zero width, zero length, zero height. These rows are removed as these are just around 20 in count and there is no possibility of diamond being present  with zero value for width, length, height.
* An attempt is made to come up with derived feature Volume as multiplication of length, width and height of diamond, but that didn't result in improvement of model performance significantly.
* It is identified that there is not much linear relationship between dependent variables and target variable and hence not much weightage is given to linear regression models.
* carat,price,x (Premium),z (Very Good),y (Good) columns are highly correlated with each other. Refer correlation plots under **docs** folder.  
* Though outliers are observed, no attempt is made either to cap them or remove them, as there is no business knowledge around these outliers to find whether these are genuine or entered by mistake/error entry

| Attribute | Number of Outliers | Percentage | Skew  |
|-----------|--------------------|------------|-------|
| price     | 3532               | 6.55       | 1.62  |
| carat     | 1883               | 3.49       | 1.12  |
| depth     | 2543               | 4.72       | -0.08 |
| table     | 604                | 1.12       | 0.8   |
| width     | 24                 | 0.04       | 0.4   |
| height    | 29                 | 0.05       | 1.58  |
| length    | 22                 | 0.04       | 2.46  |

## Models
* 20% of dataset is used to test models
* Almost all available regression models are evaluated to identify how they are performing against dataset. R2 Score and RMSE metrics are calculated to evaluate models.Cross validation is performed for 10 splits and average R2 Score and RMSE metrics calculated for both test and train data to ensure better generalization of model. All these metrics are available as json files under Model\GridSearchCV\Untuned folder.
* Out of all evaluated models **CatBoost Regressor** outperformed other models with better R2 score and with minimal difference in RMSE between Train and Test data.
* Hyperparameter tuning is performed for top 2 (in terms of R2 score and RMSE metrics) regression models CatBoost & LightGBM by making use of **optuna**, unfortunately no further improvement in performance is observed for these two regression models.
* Tried out Artificial Neural Networks as well, but unfortunately unable to proceed further with cross validation as it was resulting in an error.

## Model Performance Metrics
| Regression Model      | R2_Test_Mean | R2_Train_Mean | RMSE_Test_Mean | RMSE_Train_Mean |
|-----------------------|--------------|---------------|----------------|-----------------|
| AdaBoostRegressor     | 0.9147       | 0.9159        | 1163.8         | 1156.18         |
| CatBoostRegressor     | 0.9829       | 0.9876        | 520.58         | 442.73          |
| DecisionTreeRegressor | 0.9657       | 0.9999        | 737.73         | 6.86            |
| ElasticNet            | 0.845        | 0.8453        | 1570.62        | 1569.65         |
| KNeighborsRegressor   | 0.9715       | 0.9999        | 673.24         | 6.78            |
| Lasso                 | 0.9012       | 0.9058        | 1251.09        | 1224.68         |
| LGBMRegressor         | 0.982        | 0.9864        | 533.19         | 465.03          |
| LinearRegression      | 0.9013       | 0.9058        | 1249.67        | 1224.65         |
| LinearSVR             | 0.8677       | 0.8677        | 1451.31        | 1451.43         |
| PolynomialFeatures    | 0.8947       | 0.9058        | 1282.81        | 1224.46         |
| RandomForestRegressor | 0.9815       | 0.9974        | 542.21         | 202.71          |
| Ridge                 | 0.9008       | 0.9058        | 1254.83        | 1224.67         |
| XGBRegressor          | 0.9812       | 0.9907        | 546.09         | 383.27          |

## Usage
* After setting up of FastAPI and StreamLit applications you can evaluate model performance by entering Diamond features and trying to predict Price for those input features.
* You can test FastAPI and StreamLit applications at below urls as well.
    * FastAPI-https://sds-cp023-diamond-price-predictor-bfar.onrender.com/docs
    * Streamlit App - https://sds-cp023-diamond-price-predictor-ui.onrender.com/

## Contact Information
https://www.linkedin.com/in/pavan-kumar-reddy-kathi-203563154/

