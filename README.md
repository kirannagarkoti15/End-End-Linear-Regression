# **End-to-End Linear Regression - Vehicle Price Prediction**  

## **Problem Statement**  
This project aims to predict **vehicle prices** based on various features using **Linear Regression**. The dataset contains vehicle-related attributes, and we apply feature engineering, preprocessing, and machine learning techniques to build an optimal prediction model.  

---

## **Getting Started**  

### **Install Dependencies**  
Before running the code, install all required dependencies using:
```bash
pip install -r requirements.txt
```
---

### **Repository Structure**

```plaintext
END-END-LINEAR-REGRESSION/
│── notebook/                  # Jupyter notebooks for step-by-step implementation
│   ├── car_price_model.ipynb         # Training notebook
│   ├── car_price_prediction.ipynb    # Prediction notebook
│   ├── boxcox_lambda_values.joblib   # Box-Cox transformation values
│   ├── final_model_lr.joblib         # Saved Linear Regression model
│   ├── scaler.joblib                 # Standardization scaler
│   ├── feature_list.json             # Selected features for modeling
│
│── output/                    # Predictions on new unseen data
│   ├── prediction.csv
│
│── processed_data/             # Preprocessed data before modeling
│   ├── processed_data.csv
│
│── raw/                        # Raw dataset files
│   ├── car data.csv
│   ├── new car data.csv
│
│── saved_models/               # Trained models and transformations
│   ├── boxcox_lambda_values.joblib
│   ├── final_model_lr.joblib
│   ├── scaler.joblib
│
│── src/                        # Modular Python scripts for production-ready use
│   ├── data_preprocessing.py    # Handles missing values, outliers, transformations
│   ├── load_configuration.py    # Loads config.yaml settings
│   ├── model_build.py           # Builds and trains the regression model
│   ├── prediction.py            # Generates predictions on new data
│ 
├── config.yaml                 # Configuration file for easy parameter changes
├── main.py                     # **Main script to execute entire pipeline**│
│── README.md                   # This file
│── requirements.txt             # Required dependencies
```
---
### **How to Use**

### Option 1: Running Jupyter Notebooks (Independent of `src/`)  
If you prefer a step-by-step interactive approach, use the Jupyter Notebooks inside the `notebook/` folder.  

1. **Train the Model**  
   - Open and run [`car_price_model.ipynb`](notebook/car_price_model.ipynb) to train the model.  

2. **Make Predictions**  
   - Open and run [`car_price_prediction.ipynb`](notebook/car_price_prediction.ipynb) to make predictions on new data.  

**Note:** All generated outputs will be saved within the same `notebook/` folder.

### Option 2: Running the Modular Pipeline (Production Ready)  

For a structured, modular approach, execute the pipeline using:  

```bash
python main.py
