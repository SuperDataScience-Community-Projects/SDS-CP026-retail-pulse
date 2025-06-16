# Retail Pulse - Customer Journey Prediction & Marketing Impact Analysis

## Description
Retail Pulse is a machine learning project designed to extract actionable insights from 10 months of real-world mobile sales data collected by TechCorner, a retail store in Rangamati, Bangladesh. The primary focus is to understand and predict customer journeys by

  * Distinguishing between new and returning buyers
  * Evaluating the effectiveness of Facebook marketing
  * Segmenting customers based on behavioral patterns

By leveraging classification, clustering, and visual analytics, this project aims to deliver a comprehensive view of customer engagement, digital marketing performance, and purchasing trends.

Link to Dataset: https://www.kaggle.com/datasets/shohinurpervezshohan/techcorner-mobile-purchase-and-engagement-data

## Project Structure

| Folder/File Name                    | Description                                                                           |
|-------------------------------------|---------------------------------------------------------------------------------------|
| api                                 | Contains FASTAPI code that host final machine learning models as rest api             |
| data_analysis                       | Contains ipynb files with EDA and cluster analysis                                    |
| frontend                            | Streamlit application the enables end users to try out these models via Fast API      |
| auto-machine-learning.py            | Contains experimenation code with AutoML for Facebook marketing model                 |
| dataset_module.py                   | Contains code to load dataset from csv file, along with few reusable transformations  |
| facebook_marketing_model.py         | Contains Final prediction model (CatBoost) implementation for facebook marketing      |
| kmeans_clustering.py                | Contains KMeans Model implementation with four clusters                               |
| optuna_hyper_parameter_tuning.ipynb | Contains hyper parameter tuning code for various machine learning models using optuna |
| returning_customer_model.py         | Contains Final prediction model (XgBoost) implementation for returning customers      |
| TechCorner_Sales_update.csv         | Dataset containing sales information for mobiles                                      |

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

| Column Name                            | Description                                                                                 |
|----------------------------------------|---------------------------------------------------------------------------------------------|
| Cus.ID                                 | Customer Id                                                                                 |
| Date                                   | Mobile Purchase date                                                                        |
| Cus. Location                          | Customer Location                                                                           |
| Age                                    | Customer Age                                                                                |
| Gender                                 | Customer Gender                                                                             |
| Mobile Name                            | Mobile Name/Model                                                                           |
| Sell Price                             | Purchase Price of Mobile                                                                    |
| Does he/she Come from Facebook Page?   | Target variable that indicates whether purchase is originated from Facebook                 |
| Does he/she Followed Our Page?         | Indicates whether customer is follower of Mobile shop Facebook page                         |
| Did he/she buy any mobile before?      | Target Variable that indicates whether customer is returning customer or brand new customer |
| Did he/she hear of our shop before?    | Indicates whether customer is already aware about mobile shop or not                        |

## Exploratory Data Analysis
* There is no missing data present in dataset
* Though outliers are observed (Sell Price Column), no attempt is made either to cap them or remove them, as there is no business knowledge around these outliers to find whether these are genuine or entered by mistake/error entry. **RobustScaler** is used for scaling purpose as it is less sensitive to outliers. There also an option to apply **boxcox** transformation on **Sell Price** Column to reduce skew from 6.06 to 0.02
* This dataset is moderately imbalanced, especially in case of target variables (Did he/she buy any mobile before? & Does he/she Come from Facebook Page?) where we can observe class label **No** is around 3 times as that of **Yes**
* **Chi-Square Test** is performed to understand relation between dependent and independent categorical variables.As all of the P values are > 0.05, There is no strong association or dependency between categorical features and target variables(Did he/she buy any mobile before? & Does he/she Come from Facebook Page?)
* Detailed Analysis results of EDA can be found at [EDA](data-analysis/retail_pulse_data_analysis.ipynb) .

## Cluster Analysis
* Sale Price seems to be  primary differentiating factor and Age is secondary differentiating factor between clusters.
* Total four clusters are suggested by Elbow method. Please note that cluster ids indicated below may vary across runs.
* Clusters 0 & 2 got same price range(12k-35k), and the differentiating factor between these clusters is age. cluster 2(**Budget-Conscious Youngsters**) got age range from 18-34 where as age range in cluster 0(**Practical Professionals**) is 34-50.
* Cluster 1 got high price range(**Elite Tech Aficionados**), cluster 3 (**Ambitious Tech Enthusiasts**) got medium price range, where as clusters 0 and 2 got starting price range.
* Detailed Analysis results of Clusters can be found at [Clusters](data-analysis/kmeans_clustering_and_analysis.ipynb)

## Models & Hyper Parameter Tuning
* 20% of dataset is used to test models
* Almost all Major classification models are evaluated to identify how they are performing against dataset.
* As Dataset is Imbalanced priority is given to F1 score. Hyper Parameter tuning is done (using optuna) to identify parameter values corresponding to maximum F1 score. These parameter values are used to get corresponding accuracy for these models.
* As this is imabalanced data set, we used StratifiedKFold validation, to preserve same class distribution as that of full dataset.
* Out of all evaluated models **CatBoostClassifier** outperformed other models with accuracy score of 0.5166 and F1 score of 0.4257 while predicting target variable Does he/she Come from Facebook Page?.
* In case of Returning Customer Prediction (Did he/she buy any mobile before?) **XGBClassifier** outperformed other models with accuracy score of 0.5357 and F1 score of 0.3651
* Additional efforts are made (with not much success) to improve F1 score and accuracy by
    * Stacking top performing models using **StackingClassifier**
    * Varying probability thresholds between 0.5 to 0.6
    * Playing around with SMOTE and Under Sampling methods
    * Making use of [AutoML](auto-machine-learning.py)
* Hyper Parameter Tuning code can be found at [Tuning](optuna_hyper_parameter_tuning.ipynb)   

## Model Performance Metrics (Returning Customer Prediction Model)
| Classification Model    | F1 score     | Accuracy Score |
|-------------------------|--------------|----------------|
| XGBClassifier           | 0.3451       | 0.5114         |
| LogisticRegression      | 0.3540       | 0.4339         |
| KNeighborsClassifier    | 0.1537       | 0.6957         |
| LinearSVC               | 0.3422       | 0.5146         |
| SVC                     | 0.3965       | 0.2473         |
| RandomForestClassifier  | 0.3506       | 0.4732         |
| DecisionTreeClassifier  | 0.3965       | 0.2473         |
| LGBMClassifier          | 0.3347       | 0.5287         |
| CatBoostClassifier      | 0.3471       | 0.5119         |

## Usage
* After setting up of FastAPI and StreamLit applications you can evaluate model performance by entering Mobile Purchase information and trying to predict target variables or cluster information.
* You can test FastAPI and StreamLit applications at below urls as well. Please note that as deployment account is Free account it may take few minutes to load these Urls. 
    * FastAPI - https://sds-cp026-retail-pulse-4clt.onrender.com/docs
    * Streamlit App - https://sds-cp026-retail-pulse-ui.onrender.com/

## Contact Information
https://www.linkedin.com/in/pavan-kumar-reddy-kathi-203563154/

