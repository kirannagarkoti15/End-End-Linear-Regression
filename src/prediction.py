import os
import pandas as pd
import numpy as np
from datetime import datetime
from src.load_configuration import configuration
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import scipy.stats as stats
from scipy.special import inv_boxcox
import json
import joblib


class PredictionOnNewData:
    def __init__(self):
        self.config = configuration().load_config()
        self.loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.numeric_features = self.config['data']['numeric_features']
        self.curr_year = datetime.now().year
        self.model_features = self.config['model']['features']
        self.model_path = os.path.join(self.loc, "saved_models", "final_model_lr.joblib")
        self.lambda_path = os.path.join(self.loc, "saved_models", "boxcox_lambda_values.joblib")
        self.scaler_path = os.path.join(self.loc, "saved_models", "scaler.joblib")
        self.prediction = os.path.join(self.loc, "output", "prediction.csv")


    def read_new_df(self):
        data = pd.read_csv(os.path.join(self.loc, "raw", "new car data.csv"))
        return data
    
    def load_saved_models(self):
        model = joblib.load(self.model_path )
        fitted_lambda = joblib.load(self.lambda_path)
        scaler = joblib.load(self.scaler_path)
        return model, fitted_lambda, scaler
    
    def feature_creation(self, data):
        data = pd.get_dummies(data, columns=['Fuel_Type', 'Seller_Type', 'Transmission'], drop_first=True)
        data['Car_Age'] = self.curr_year - data['Year']
        for col in self.model_features :
            if col not in data.columns:
                data[col] = 0 
        return data
        
    def box_cox_transformation(self, data, fitted_lambda):
        data_transformed = data.copy()
        for col in self.numeric_features:
            data_transformed[col] = stats.boxcox(data_transformed[col] + 1, fitted_lambda[col])
        data_transformed = data_transformed[self.model_features]
        return data_transformed
    
    def get_prediction(self):
        data = self.read_new_df()
        model, fitted_lambda, scaler = self.load_saved_models()
        data = self.feature_creation(data)
        data.drop(columns=['Car_Name', 'Year', 'Fuel_Type_Petrol'], inplace=True)
        data_transformed = self.box_cox_transformation(data, fitted_lambda)
        data_scaled = scaler.transform(data_transformed)
        predicted_price_transformed = model.predict(data_scaled)
        predicted_price_transformed = predicted_price_transformed.reshape(-1, 1)
        predicted_prices_original = inv_boxcox(predicted_price_transformed, fitted_lambda['Selling_Price']) - 0.1
        data["Predicted_Selling_Price"] = predicted_prices_original
        data.to_csv(self.prediction)
        return data