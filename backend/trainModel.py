import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import re
from bs4 import BeautifulSoup
import tldextract

# Feature extraction function
def extract_features(url, html_content):
    features = []
    
    # 1. URL Length
    url_length = len(url)
    features.append(url_length)
    
    # 2. Number of Special Characters
    special_chars = len(re.findall(r'[?&=]', url))
    features.append(special_chars)
    
    # 3. Use of IP Address in URL
    use_ip = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', url) else 0
    features.append(use_ip)
    
    # 4. Presence of HTTPS
    https = 1 if url.startswith('https://') else 0
    features.append(https)
    
    # Parse HTML content
    soup = BeautifulSoup(html_content, 'lxml')
    
    # 5. Number of Hidden Elements
    hidden_elements = len(soup.find_all('input', type='hidden'))
    features.append(hidden_elements)
    
    # 6. Number of External Links
    external_links = len([a for a in soup.find_all('a', href=True) if 'http' in a['href'] and tldextract.extract(a['href']).domain != tldextract.extract(url).domain])
    features.append(external_links)
    
    # 7. Domain Age in Days (Placeholder)
    domain_age_days = 365  # Example placeholder
    features.append(domain_age_days)
    
    # 8. Domain Reputation Score (Placeholder)
    domain_reputation_score = 50  # Example placeholder
    features.append(domain_reputation_score)
    
    return features

# Load the dataset
data = pd.read_csv('phishing_dataset.csv')

# Apply the feature extraction function to each row
data['features'] = data.apply(lambda row: extract_features(row['url'], row['html_content']), axis=1)

# Split the features into separate columns
features_df = pd.DataFrame(data['features'].tolist(), columns=[
    'url_length', 'num_special_chars', 'use_ip', 'https',
    'num_hidden_elements', 'num_external_links',
    'domain_age_days', 'domain_reputation_score'
])

# Combine the features with the target column
final_data = pd.concat([features_df, data['is_phishing']], axis=1)

# Save the processed data to a new CSV (optional)
final_data.to_csv('processed_phishing_dataset.csv', index=False)

# Separate features and target variable
X = final_data.drop('is_phishing', axis=1)
y = final_data['is_phishing']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model (optional)
accuracy = model.score(X_test, y_test)
print(f'Model accuracy: {accuracy:.2f}')

# Save the trained model
joblib.dump(model, 'phishing_model.pkl')
