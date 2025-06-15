import pandas as pd
import dataset_module as ds
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from flaml import AutoML
from sklearn.metrics import accuracy_score

# Try out AutoML, to see if it can provide better accuracy over 65%
pd.set_option('display.max_columns',None)

# Load data set
dataset = ds.get_data_frame(False, False)
# print(dataset.columns)

# Separate dependent and independent variables
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# Split data into Training and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=0)
# print(X_train)

# Apply Standard/Robust Scaling along with One hot encoding for Mobile Model
categorical_features = ['mobile_model']
standard_scaling_features = ['age']
robust_scaling_features = ['price']
# , drop='first'
preprocessor = ColumnTransformer([('onehotencoder', OneHotEncoder(handle_unknown='ignore'), categorical_features),
                                  ('standardscaler', StandardScaler(), standard_scaling_features)
                                  ,('robustscaler', RobustScaler(), robust_scaling_features)
                                  ], remainder='passthrough')

pipeline = Pipeline([('preprocessor', preprocessor)])
# print(X_train)
X_train_transformed = pipeline.fit_transform(X_train)
X_test_transformed = pipeline.transform(X_test)

automl = AutoML()
automl_settings = {
    # "time_budget": 60,  # Time in seconds
    "metric": "accuracy",
    "task": "classification",
}
automl.fit(X_train_transformed, y_train, automl_settings)
print(automl.model)
print(automl.best_config)
print(accuracy_score(y_test, automl.predict(X_test_transformed)))
