from flask import Flask, request, jsonify
import joblib
import pandas as pd
import re
from bs4 import BeautifulSoup

# Load the trained model
model = joblib.load('phishing_model.pkl')

# Initialize Flask app
app = Flask(__name__)

# Feature extraction function
def extract_features(url, html_content):
    features = []
    features.append(len(url))
    features.append(len(re.findall(r'[?&=]', url)))
    features.append(1 if re.match(r'\d+\.\d+\.\d+\.\d+', url) else 0)
    features.append(1 if url.startswith('https://') else 0)
    
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    features.append(len(soup.find_all('input', type='hidden')))
    features.append(len([a for a in soup.find_all('a', href=True) if 'http' in a['href'] and tldextract.extract(a['href']).domain != tldextract.extract(url).domain]))
    features.append(365)  # Placeholder for domain age in days
    features.append(50)   # Placeholder for domain reputation score
    
    return features

@app.route('/')
def index():
    return "Phishing Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get the JSON data
    url = data['url']  # Extract the URL from the JSON data
    html_content = data['html_content']  # Extract the HTML content from the JSON data

    # Extract features from the URL and HTML content
    features = extract_features(url, html_content)
    
    # Convert to DataFrame for model prediction
    df = pd.DataFrame([features])
    
    # Predict using the model
    prediction = model.predict(df)[0]
    
    # Return the prediction
    result = {'is_phishing': bool(prediction)}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
