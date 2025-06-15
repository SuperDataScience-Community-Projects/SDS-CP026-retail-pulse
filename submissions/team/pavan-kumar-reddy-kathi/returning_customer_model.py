import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import dataset_module as ds
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib
from pathlib import Path

# Final prediction model for Returning Customer, after hyper parameter tuning
pd.set_option('display.max_columns', None)
# Load data set
dataset = ds.get_data_frame(False, True)
print(dataset.columns)

# # Separate dependent and independent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split data into Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
print(len(y[y == 0]))
print(len(y[y == 1]))

# Apply Standard/Robust Scaling/one hot encoding
standard_scaling_features = [0] # Age
robust_scaling_features = [1] # Price
categorical_features = [2] # Mobile Model

preprocessor = ColumnTransformer([
    ('standardscaler', StandardScaler(), standard_scaling_features)
    , ('robustscaler', RobustScaler(), robust_scaling_features)
    ,('onehotencoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

params_xgb={'max_depth': 2, 'learning_rate': 0.015338655903040864, 'n_estimators': 53, 'subsample': 0.7260901466089422,
            'colsample_bytree': 0.9330173253181531,'scale_pos_weight': len(y[y == 0]) / len(y[y == 1])}

# Fit model and make predictions
pipeline = Pipeline([('preprocessor', preprocessor),
                     ('classification_model',XGBClassifier(**params_xgb))])
pipeline.fit(X_train, y_train)
y_predict = pipeline.predict(X_test)

# Classification metrics
print(accuracy_score(y_test, y_predict))
print(f1_score(y_test, y_predict))
print(precision_score(y_test, y_predict))
print(recall_score(y_test, y_predict))

# Save Model
path_to_save = Path.cwd()/'api/models/returning_customer.pkl'
with open(path_to_save, 'wb') as file:
    joblib.dump(pipeline, file)


