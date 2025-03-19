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


class ModelBuild:
    def __init__(self):
        self.config = configuration().load_config()
        self.loc = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.numeric_features = self.config['data']['numeric_features']
        self.target = self.config['data']['target']
        self.model_path = os.path.join(self.loc, "saved_models", "final_model_lr.joblib")
        self.lambda_path = os.path.join(self.loc, "saved_models", "boxcox_lambda_values.joblib")
        self.scaler_path = os.path.join(self.loc, "saved_models", "scaler.joblib")


    def read_processed_df(self):
        data = pd.read_csv(os.path.join(self.loc, "processed_data", "processed_data.csv"))
        return data
    
    def box_cox_transformation(self, data):
        X = data.drop(columns=[self.target])
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        fitted_lambda = pd.Series(np.zeros(len(self.numeric_features), dtype=np.float64), index=self.numeric_features)
        y_train, fitted_lambda[self.target] = stats.boxcox(y_train + 0.1)
        for col in X_train[self.numeric_features]:
            X_train[col], fitted_lambda[col] = stats.boxcox(X_train[col] + 1)
        y_test = stats.boxcox(y_test + 0.1, fitted_lambda[self.target])
        for col in X_test[self.numeric_features]:
            X_test[col] = stats.boxcox(X_test[col] + 1, fitted_lambda[col])
        y_train = pd.DataFrame(y_train, index=X_train.index, columns=[self.target])
        y_test = pd.DataFrame(y_test, index=X_test.index, columns=[self.target])
        X_boxcox = pd.concat([X_train, X_test])
        y_boxcox = pd.concat([y_train, y_test])
        df_boxcox = pd.concat([X_boxcox, y_boxcox], axis=1)
        df_boxcox.sort_index(inplace=True)
        return fitted_lambda, df_boxcox
    
    def get_train_test_split(self, data):
        X = data.drop(columns=[self.target])
        y = data[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=19)
        return X_train, X_test, y_train, y_test 
    
    def statndardization(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return scaler, X_train_scaled, X_test_scaled
    
    def evaluate_model(self, y_true, y_pred, dataset_type="Test"):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f"\n {dataset_type} Set Performance:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R-squared (R²): {r2:.2f}")

    def save_models(self, model, fitted_lambda, scaler):
        joblib.dump(model, self.model_path)
        joblib.dump(fitted_lambda, self.lambda_path)
        joblib.dump(scaler, self.scaler_path)
    
    def train_model(self):
        data = self.read_processed_df()
        fitted_lambda, df_boxcox = self.box_cox_transformation(data)
        X_train, X_test, y_train, y_test  = self.get_train_test_split(df_boxcox)
        scaler, X_train_scaled, X_test_scaled = self.statndardization(X_train, X_test)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        self.evaluate_model(y_train, y_train_pred, "Train")
        self. evaluate_model(y_test, y_test_pred, "Test")
        self.save_models(model, fitted_lambda, scaler)
        return model