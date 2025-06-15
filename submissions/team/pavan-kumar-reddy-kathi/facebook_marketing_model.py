import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import dataset_module as ds
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import joblib
from pathlib import Path

# Final prediction model for facebook marketing, after hyper parameter tuning
pd.set_option('display.max_columns', None)

# Load data set
dataset = ds.get_data_frame(False, False)
print(dataset.columns)

# Separate dependent and independent variables
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Split data into Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

# Apply Standard/Robust Scaling
standard_scaling_features = [0] # Age
robust_scaling_features = [1] # Price

preprocessor = ColumnTransformer([
    ('standardscaler', StandardScaler(), standard_scaling_features)
    , ('robustscaler', RobustScaler(), robust_scaling_features)
], remainder='passthrough')

# Fit model and make predictions
pipeline = Pipeline([('preprocessor', preprocessor),
                     ('classification_model',CatBoostClassifier(iterations=224, learning_rate=0.0260788345975822,
                             depth=5,  cat_features=[2],auto_class_weights='Balanced', verbose=False))])
pipeline.fit(X_train, y_train)
y_predict = pipeline.predict(X_test)

# Classification metrics
print(accuracy_score(y_test, y_predict))
print(f1_score(y_test, y_predict))
print(precision_score(y_test, y_predict))
print(recall_score(y_test, y_predict))

# Save Model
path_to_save = Path.cwd()/'api/models/facebook_marketing.pkl'
with open(path_to_save, 'wb') as file:
    joblib.dump(pipeline, file)


