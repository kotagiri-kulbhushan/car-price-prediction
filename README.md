Car Price Prediction 🚗


Live Demo : https://car-price-prediction-p614.onrender.com/


Project Overview

Car Price Prediction is a web application that predicts car selling prices based on various car features such as year, present price, kilometers driven, fuel type, seller type, transmission, and owner history. 
The app uses a Machine Learning model trained on real car sales data to generate accurate price predictions.
This project demonstrates end-to-end ML deployment with a Flask backend and a scalable cloud deployment on Render.

Features
Predict car prices instantly by filling a simple form
Input parameters: Year, Present Price, Kms Driven, Fuel Type, Seller Type, Transmission, Owner
Responsive web design for desktop and mobile devices
Machine Learning model trained on real car dataset
Preprocessing pipeline with StandardScaler and OneHotEncoder
Deployed on Render with Gunicorn WSGI server

Backend & ML
Framework: Flask
Machine Learning: scikit-learn
Data Processing: pandas, numpy
Model Persistence: joblib
WSGI Server: Gunicorn

Frontend
HTML5 with responsive design
CSS3 for styling

Deployment
Platform: Render

Usage
Access the web application
Fill in the car details:
Present Price (in Lakhs)
Manufacturing Year
Kilometers Driven
Fuel Type (Petrol/Diesel/CNG)
Seller Type (Dealer/Individual)
Transmission Type (Manual/Automatic)
Number of Previous Owners
Click "Predict Price" to get the estimated selling price


Model Details

Algorithm
Random Forest Regressor with optimized hyperparameters
R² Score: ~0.85 (on test data)

Preprocessing
Numerical Features: StandardScaler (Year, Present_Price, Kms_Driven)
Categorical Features: OneHotEncoder (Fuel_Type, Seller_Type, Transmission, Owner)
Feature Engineering: Pipeline-based preprocessing


