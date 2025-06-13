import data.jeju_traffic_data as jt

train, test = jt.load(data_folder="processed")

# train = train.sample(1000000)
# test = test.sample(100000)

# Support Vector Regression Model
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numerical_features = ['start_latitude', 'start_longitude', 'end_latitude', 'end_longitude', 'year', 
                      'month', 'day', 'cos_time', 'sin_time', 'base_hour', 
                      'maximum_speed_limit', 'lane_count', 'hour_mean_target', 'whs_mean_target', 
                      'multi_speed_penalty', 'speed_weight']
categorical_features = train.columns.to_list()
for n in numerical_features:
  categorical_features.remove(n)
categorical_features.remove('target')

X_train = train.drop('target', axis=1)
y_train = train['target']

X_test = test.drop('target', axis=1)
y_test = test['target']

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_features)
])

svr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", SVR(kernel='rbf', C=1.0, epsilon=0.2))
])

# 학습 및 예측
svr_model.fit(X_train, y_train)
y_svr_pred = svr_model.predict(X_test)

# 성능 평가
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_svr_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_svr_pred)
r2 = r2_score(y_test, y_svr_pred)

# 출력
print("MSE  : %.4f" %mse)
print("RMSE : %.4f" %rmse)
print("MAE  : %.4f" %mae)
print("R²   : %.4f" %r2)