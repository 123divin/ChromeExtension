from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('phishing_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define a route for the default URL, which loads the form
@app.route('/')
def index():
    return "Phishing Detection API"

# Define a route for the prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data
    url_features = data['features']  # Assume features are passed as a JSON key 'features'

    # Convert to DataFrame for model prediction
    df = pd.DataFrame([url_features])
    
    # Predict using the model
    prediction = model.predict(df)[0]
    
    # Return the prediction
    result = {'is_phishing': bool(prediction)}
    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
