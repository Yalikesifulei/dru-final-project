import pickle
import json
import pandas as pd

from utils.dataloader import DataLoader 
from settings.constants import TRAIN_CSV, SAVED_ESTIMATOR
from xgboost import XGBRegressor


with open('settings/specifications.json') as f:
    specifications = json.load(f)

raw_train = pd.read_csv(TRAIN_CSV)
x_columns = specifications['description']['X']
y_column = specifications['description']['y']

X_raw = raw_train[x_columns]

loader = DataLoader()
loader.fit(X_raw)
X = loader.load_data()
y = raw_train['SalePrice']

params = {'learning_rate': 0.025, 'max_depth': 3, 'n_estimators': 2048}
model = XGBRegressor(**params)
model.fit(X, y)
with open(SAVED_ESTIMATOR, 'wb')as f:
    pickle.dump(model, f)