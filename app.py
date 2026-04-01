from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

app = Flask(__name__)

# Constants
MODEL_PATH = "model.pkl"
DATASET_PATH = "dataset.csv"

# Global model and data variable
model = None
df = None

def load_resources():
    global model, df
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)

load_resources()

def retrain_model_internal():
    global model, df
    if df is None:
        return None
    
    # Preprocessing
    df_clean = df.dropna().drop_duplicates()
    X = df_clean.drop(columns=['price'])
    y = df_clean['price']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Categorical features
    cat_cols = ['company', 'model', 'fuel_type', 'transmission', 'owner_type']
    
    # Fitting encoder
    ohe = OneHotEncoder()
    ohe.fit(X[cat_cols])
    
    # Pipeline
    column_transformer = make_column_transformer(
        (OneHotEncoder(categories=ohe.categories_), cat_cols),
        remainder='passthrough'
    )
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = make_pipeline(column_transformer, rf_model)
    
    pipeline.fit(X_train, y_train)
    
    # Save 
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    
    model = pipeline
    return r2_score(y_test, pipeline.predict(X_test))


@app.route('/')
def index():
    if df is None:
        return "Dataset not found. Please upload dataset.csv.", 404
    
    companies = sorted(df['company'].unique())
    years = sorted(df['year'].unique(), reverse=True)
    fuel_types = sorted(df['fuel_type'].unique())
    transmissions = sorted(df['transmission'].unique())
    owners = sorted(df['owner_type'].unique())
    
    return render_template(
        'ind.html',
        companies=companies,
        years=years,
        fuel_types=fuel_types,
        transmissions=transmissions,
        owners=owners
    )

@app.route('/get_models/<company>')
def get_models(company):
    if df is None:
        return jsonify([])
    models = sorted(df[df['company'] == company]['model'].unique())
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        company = data.get('company')
        model_name = data.get('model')
        year = int(data.get('year'))
        fuel_type = data.get('fuel_type')
        transmission = data.get('transmission')
        km_driven = int(data.get('km_driven'))
        owner_type = data.get('owner_type')
        mileage = float(data.get('mileage'))
        engine = int(data.get('engine'))

        # Prepare for prediction
        input_df = pd.DataFrame([[company, model_name, year, fuel_type, transmission, km_driven, owner_type, mileage, engine]],
                               columns=['company', 'model', 'year', 'fuel_type', 'transmission', 'km_driven', 'owner_type', 'mileage', 'engine'])
        
        prediction = model.predict(input_df)[0]
        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/upload', methods=['POST'])
def upload_dataset():
    global df
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        file.save(DATASET_PATH)
        df = pd.read_csv(DATASET_PATH)
        score = retrain_model_internal()
        return jsonify({
            "message": "Dataset uploaded and model retrained successfully!",
            "r2_score": round(score, 4)
        })
    
    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
