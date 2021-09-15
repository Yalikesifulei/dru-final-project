from xgboost import XGBRegressor

class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        params = {'learning_rate': 0.025, 'max_depth': 3, 'n_estimators': 2048}
        return XGBRegressor(**params).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)