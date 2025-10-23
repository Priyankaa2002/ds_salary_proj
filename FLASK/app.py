from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)

# ------------------------------------------------------------
# 1. LOAD MODEL AND TRAINING FEATURE STRUCTURE
# ------------------------------------------------------------
MODEL_PATH = "model_file.p"
DATA_PATH = "eda_data.csv"

# Load the trained model
with open(MODEL_PATH, "rb") as file:
    model_data = pickle.load(file)
    model = model_data["model"]

# Load dataset to rebuild feature columns
df = pd.read_csv(DATA_PATH)
df_model = df[['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry',
               'Sector', 'Revenue', 'num_comp', 'hourly', 'employer_provided',
               'job_state', 'same_state', 'age', 'python_yn', 'spark', 'aws',
               'excel', 'job_simp', 'seniority', 'desc_len']]

df_dum = pd.get_dummies(df_model, drop_first=True)
feature_columns = df_dum.drop("avg_salary", axis=1).columns

# ------------------------------------------------------------
# 2. HELPER FUNCTION TO PROCESS NEW INPUTS
# ------------------------------------------------------------
def preprocess_input(input_dict):
    """
    Convert incoming JSON into a DataFrame that matches
    the one-hot encoded training structure.
    """
    # Create a one-row DataFrame from input
    input_df = pd.DataFrame([input_dict])

    # One-hot encode it
    input_dummies = pd.get_dummies(input_df)

    # Reindex to match training columns (fill missing with 0)
    input_dummies = input_dummies.reindex(columns=feature_columns, fill_value=0)

    return input_dummies

# ------------------------------------------------------------
# 3. DEFINE PREDICTION ROUTE
# ------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request
        request_json = request.get_json()

        # Preprocess user input
        x_input = preprocess_input(request_json)

        # Make prediction
        prediction = model.predict(x_input)[0]

        # Return prediction as JSON
        return jsonify({'predicted_avg_salary': round(float(prediction), 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ------------------------------------------------------------
# 4. RUN APP
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
