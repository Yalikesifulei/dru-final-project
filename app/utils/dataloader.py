import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # new variable
        self.dataset['1st_2ndFlrSF_temp'] = self.dataset['1stFlrSF'] + self.dataset['2ndFlrSF']
        self.dataset['TotalSF'] = self.dataset['TotalBsmtSF'] + self.dataset['1st_2ndFlrSF_temp']
        # self.dataset['TotalSF'] = self.dataset['TotalBsmtSF'] + self.dataset['1stFlrSF'] + self.dataset['2ndFlrSF']

        # drop index and temp column
        self.dataset = self.dataset.drop(['Id', '1st_2ndFlrSF_temp'], axis=1)

        # LabelEncoder for categorical
        cat_vars = [
            "MSSubClass",
            "MSZoning",
            "Street",
            "Alley",
            "LotShape",
            "LandContour",
            "Utilities",
            "LotConfig",
            "LandSlope",
            "Neighborhood",
            "Condition1",
            "Condition2",
            "BldgType",
            "HouseStyle",
            "OverallQual",
            "OverallCond",
            "RoofStyle",
            "RoofMatl",
            "Exterior1st",
            "Exterior2nd",
            "MasVnrType",
            "ExterQual",
            "ExterCond",
            "Foundation",
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "BsmtFinType1",
            "BsmtFinType2",
            "Heating",
            "HeatingQC",
            "CentralAir",
            "Electrical",
            "KitchenQual",
            "Functional",
            "FireplaceQu",
            "GarageType",
            "GarageFinish",
            "GarageQual",
            "GarageCond",
            "PavedDrive",
            "PoolQC",
            "Fence",
            "MiscFeature",
            "SaleCondition",
            "SaleType"
        ]
        le = LabelEncoder()
        for col in cat_vars:
            le.fit(self.dataset[col])
            self.dataset[col] = le.transform(self.dataset[col])

        return self.dataset