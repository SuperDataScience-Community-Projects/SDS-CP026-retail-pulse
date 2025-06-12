import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.cluster import KMeans
import dataset_module as ds
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib


pd.set_option('display.max_columns',None)

# Load data set
dataset = ds.get_data_frame(False, True)
print(dataset.columns)

# Apply Standard/Robust Scaling & OneHotEncoding
standard_scaling_features = [0] # Age
robust_scaling_features = [1] # Price
categorical_features = [2] # Mobile Model

preprocessor = ColumnTransformer([
    ('standardscaler', StandardScaler(), standard_scaling_features)
    , ('robustscaler', RobustScaler(), robust_scaling_features)
    ,('onehotencoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

pipeline = Pipeline([('preprocessor', preprocessor),
                     ('kmeans_model', KMeans(4, init='k-means++', n_init=10, random_state=42))])

# Fit pipeline on dataset
pipeline.fit(dataset)

# Populate cluster details & analyze clusters
# kmeans_model = pipeline.named_steps['kmeans_model']
# dataset['cluster'] = kmeans_model.labels_

# print(dataset.head())
# print(dataset.groupby('cluster').agg({'age':['min', 'max', 'mean'], 'price':['min','max','mean']}))

# Save Model
path_to_save = Path.cwd()/'api/models/customer_clusters.pkl'
with open(path_to_save, 'wb') as file:
    joblib.dump(pipeline, file)

