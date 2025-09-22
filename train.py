import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, r2_score 
from joblib import dump 
import os 
 
"# Loading the data with relative path" 
current_dir = os.path.dirname(__file__) 
data_path = os.path.join(current_dir, 'data', 'cardata.csv') 
df = pd.read_csv(data_path) 
 
print("Dataset shape:", df.shape) 
print(df.head()) 
print(df.isnull().sum()) 
 
"# Feature and target" 
X = df.drop(columns=["Selling_Price", "Car_Name"]) 
y = df["Selling_Price"] 
 
"# Train-test split" 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
"# Preprocessing" 
numerical_features = ["Year", "Present_Price", "Kms_Driven"] 
categorical_features = ["Fuel_Type", "Seller_Type", "Transmission", "Owner"] 
 
preprocessor = ColumnTransformer( 
    transformers=[ 
        ("num", StandardScaler(), numerical_features), 
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features), 
    ] 
) 
 
"# Models to compare" 
models = { 
    "Linear Regression": LinearRegression(), 
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42) 
} 
 
best_model = None 
best_score = -np.inf 
 
"# Train and evaluate models" 
for name, model in models.items(): 
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)]) 
    pipeline.fit(X_train, y_train) 
    preds = pipeline.predict(X_test) 
 
    r2 = r2_score(y_test, preds) 
    mse = mean_squared_error(y_test, preds) 
 
    print(f"{name} -> R2: {r2:.4f}, MSE: {mse:.4f}") 
 
    if r2 
        best_score = r2 
        best_model = pipeline 
 
"# Save best model with relative path" 
models_dir = os.path.join(current_dir, 'models') 
os.makedirs(models_dir, exist_ok=True) 
model_save_path = os.path.join(models_dir, 'best_model.joblib') 
dump(best_model, model_save_path) 
print("? Best model saved with Rý =", best_score) 
