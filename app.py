from flask import Flask, request, render_template 
from joblib import load 
import pandas as pd 
import os 
 
app = Flask(__name__) 
 
"# Load trained model with relative path" 
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.joblib') 
model = load(model_path) 
 
@app.route("/") 
def home(): 
    return render_template("index.html") 
 
@app.route("/predict", methods=["POST"]) 
def predict(): 
    try: 
        "# Collect input" 
        data = request.form 
        year = int(data["year"]) 
        present_price = float(data["present_price"]) 
        kms_driven = int(data["kms_driven"]) 
        fuel_type = data["fuel_type"] 
        seller_type = data["seller_type"] 
        transmission = data["transmission"] 
        owner = int(data["owner"]) 
 
        "# Create DataFrame with correct feature order" 
        df = pd.DataFrame([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]], 
            columns=["Year", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner"]) 
 
        "# Predict using trained pipeline" 
        prediction = model.predict(df)[0] 
 
        "# Render result" 
        return render_template("results.html", prediction=prediction) 
 
    except Exception as e: 
        return f"Error: {str(e)}", 400 
 
if __name__ == "__main__": 
    app.run(host='0.0.0.0', port=5000, debug=False) 
