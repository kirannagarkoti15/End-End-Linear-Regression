import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.load_configuration import configuration


class DataPrepration:
    def __init__(self):
        self.config = configuration().load_config()
        self.loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.curr_year = datetime.now().year
        self.numeric_features = [col for col in self.config['data']['numeric_features'] if col != "Owner"]


    def data_read(self):
        data = pd.read_csv(os.path.join(self.loc, "raw", "car data.csv"))
        return data

    def feature_creation(self, data):
        data = pd.get_dummies(data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)
        data['Car_Age'] = self.curr_year - data['Year']
        return data

    def outlier_removal(self, data):

        Q1 = data[self.numeric_features].quantile(0.25)
        Q3 = data[self.numeric_features].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3* IQR
        upper_bound = Q3 + 3* IQR
        mask = ~((data[self.numeric_features] < lower_bound) | (data[self.numeric_features] > upper_bound)).any(axis=1).astype(bool)
        data_processed = data[mask]
        return data_processed
    
    def data_process(self):
        data = self.data_read()
        data = self.feature_creation(data)
        data_processed = self.outlier_removal(data)
        data_processed.drop(columns=['Car_Name', 'Year', 'Fuel_Type_Petrol'], inplace=True)
        data_processed.to_csv(self.loc + '/processed_data/processed_data.csv', index=False)
        return data_processed
