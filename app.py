from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model, encoders, and scaler
try:
    with open('best_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('encoder.pkl', 'rb') as encoders_file:
        encoders = pickle.load(encoders_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler_data = pickle.load(scaler_file)
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

app = Flask(__name__)

def make_prediction(input_data):
    try:
        # Convert input data into a DataFrame
        input_df = pd.DataFrame([input_data])
        print("Input Data Received:", input_df)  # Debugging: Print input data

        # Encode categorical columns
        for col, encoder in encoders.items():
            if col in input_df.columns:  # Validate column existence
                input_df[col] = encoder.transform(input_df[col])
            else:
                raise KeyError(f"Column '{col}' missing from input data.")

        # Scale numerical columns
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        if all(col in input_df.columns for col in numerical_cols):
            input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])
        else:
            missing_cols = [col for col in numerical_cols if col not in input_df.columns]
            raise KeyError(f"Missing numerical columns: {missing_cols}")

        # Make predictions
        prediction = loaded_model.predict(input_df)[0]
        probability = loaded_model.predict_proba(input_df)[0, 1]

        # Convert prediction to "Yes" or "No"
        prediction_text = "Churn" if prediction == 1 else "No Churn"
        print(f"Prediction: {prediction_text}, Probability: {probability:.2f}")  # Debugging output

        return prediction_text, probability
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error", 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    if request.method == 'POST':
        try:
            # Collect input data from the form
            input_data = {
                'gender': request.form['gender'],
                'SeniorCitizen': int(request.form['SeniorCitizen']),
                'Partner': request.form['Partner'],
                'Dependents': request.form['Dependents'],
                'tenure': int(request.form['tenure']),
                'PhoneService': request.form['PhoneService'],
                'MultipleLines': request.form['MultipleLines'],
                'InternetService': request.form['InternetService'],
                'OnlineSecurity': request.form['OnlineSecurity'],
                'OnlineBackup': request.form['OnlineBackup'],
                'DeviceProtection': request.form['DeviceProtection'],
                'TechSupport': request.form['TechSupport'],
                'StreamingTV': request.form['StreamingTV'],
                'StreamingMovies': request.form['StreamingMovies'],
                'Contract': request.form['Contract'],
                'PaperlessBilling': request.form['PaperlessBilling'],
                'PaymentMethod': request.form['PaymentMethod'],
                'MonthlyCharges': float(request.form['MonthlyCharges']),
                'TotalCharges': float(request.form['TotalCharges']),
            }
            print("Form Data Received:", input_data)  # Debugging: Print form data

            # Call the prediction function
            prediction, probability = make_prediction(input_data)
        except Exception as e:
            print(f"Error during form handling: {e}")
            prediction = "Error"
            probability = 0.0

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=False)
