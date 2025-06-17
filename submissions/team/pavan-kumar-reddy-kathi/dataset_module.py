import calendar

import numpy as np
import pandas as pd
from scipy.stats import boxcox

updated_columns = ['customer_id', 'purchase_date', 'is_local', 'age', 'gender', 'mobile_model', 'price',
                   'is_from_facebook_page', 'is_facebook_page_follower', 'is_returning_customer',
                   'awareness_through_marketing']

def get_data_frame(apply_price_transformation:bool = False,
                   is_returning_customer_dependent_variable:bool = False)->pd.DataFrame:
    """
    Loads data from CSV file into DataFrame and returns the same.
    :return: DataFrame holding data from CSV file
    """
    dataset = pd.read_csv('TechCorner_Sales_update.csv')
    dataset.columns = updated_columns

    # Date Handling
    dataset['purchase_date'] = pd.to_datetime(dataset['purchase_date'], errors='raise', dayfirst=True)

    # Periodic Features
    dataset['day_of_year'] = dataset['purchase_date'].dt.dayofyear

    # Check leap year (366 days if leap year, else 365)
    dataset['year'] = dataset['purchase_date'].dt.year
    dataset['days_in_year'] = dataset['year'].apply(lambda x: 366 if calendar.isleap(x) else 365)

    # sine/cosine transformations
    dataset['sin_day_of_year'] = np.sin(2*np.pi*dataset['day_of_year']/dataset['days_in_year'])
    dataset['cos_day_of_year'] = np.cos(2 * np.pi * dataset['day_of_year'] / dataset['days_in_year'])

    # Drop purchase date column
    dataset.drop(columns=['customer_id', 'purchase_date', 'day_of_year', 'year', 'days_in_year'], inplace=True)


    # map yes/no to 1/0, instead of performing one hot encoding
    dataset['is_local'] = dataset['is_local'].map({'Rangamati Sadar':1, 'Inside Rangamati':1, 'Outside Rangamati':0})
    dataset['gender'] = dataset['gender'].map({'F':1, 'M':0})
    dataset['is_from_facebook_page'] = dataset['is_from_facebook_page'].map({'Yes':1, 'No':0})
    dataset['is_facebook_page_follower'] = dataset['is_facebook_page_follower'].map({'Yes':1, 'No':0})
    dataset['is_returning_customer'] = dataset['is_returning_customer'].map({'Yes':1, 'No':0})
    dataset['awareness_through_marketing'] = dataset['awareness_through_marketing'].map({'Yes':1, 'No':0})

    if apply_price_transformation:
        dataset['price'], lambda_bc = boxcox(dataset['price'])

    last_column = ['is_from_facebook_page']
    if is_returning_customer_dependent_variable:
        last_column = ['is_returning_customer']

    columns_at_start = ["age", "price", "mobile_model"]
    re_ordered_columns = (columns_at_start +
                          [col for col in dataset.columns if col not in columns_at_start + last_column] +
                          last_column)

    dataset = dataset[re_ordered_columns]
    return dataset



