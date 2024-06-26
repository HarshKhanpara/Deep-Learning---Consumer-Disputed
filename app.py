from flask import Flask, request, jsonify
import torch
import pandas as pd
import joblib
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(12, 128)  # Assuming 12 input features
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model
model = Model()
model.load_state_dict(torch.load('./model.pth', map_location=torch.device('cpu')))
model.eval()

import json
with open('encoded_values.json') as f:
    encoded_values = json.load(f)

# Load the encoder
encoder = joblib.load('encoder.pkl')

# Define the columns your model is expecting
model_columns = [
    'Product', 'Sub-product', 'Issue', 'Sub-issue', 'Company public response', 
    'Company', 'State', 'ZIP code', 'Consumer consent provided?', 
    'Submitted via', 'Company response to consumer', 'Timely response?'
]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data, index=[0])
    
    #Extract model_columns from the input data
    product = df['Product'].values[0]
    enc_product = encoded_values['Product'][product]
    df['Product'] = enc_product
    print(enc_product)
    
    sub_product = df['Sub-product'].values[0]
    enc_sub_product = encoded_values['Sub-product'][sub_product]
    df['Sub-product'] = enc_sub_product
    
    issue = df['Issue'].values[0]
    enc_issue = encoded_values['Issue'][issue]
    df['Issue'] = enc_issue
    
        # Extracting and encoding for Sub-issue
    sub_issue = df['Sub-issue'].values[0]
    enc_sub_issue = encoded_values['Sub-issue'][sub_issue]
    df['Sub-issue'] = enc_sub_issue
    
    # Encoding for Company public response
    company_public_response = df['Company public response'].values[0]
    enc_company_public_response = encoded_values['Company public response'][company_public_response]
    df['Company public response'] = enc_company_public_response
    
    # Encoding for Company
    company = df['Company'].values[0]
    enc_company = encoded_values['Company'][company]
    df['Company'] = enc_company
    
    # Encoding for State
    state = df['State'].values[0]
    enc_state = encoded_values['State'][state]
    df['State'] = enc_state
    
    # Encoding for ZIP code
    zip_code = df['ZIP code'].values[0]
    enc_zip_code = encoded_values['ZIP code'][zip_code]
    df['ZIP code'] = enc_zip_code
    
    # Encoding for Consumer consent provided?
    consent_provided = df['Consumer consent provided?'].values[0]
    enc_consent_provided = encoded_values['Consumer consent provided?'][consent_provided]
    df['Consumer consent provided?'] = enc_consent_provided
    
    # Encoding for Submitted via
    submitted_via = df['Submitted via'].values[0]
    enc_submitted_via = encoded_values['Submitted via'][submitted_via]
    df['Submitted via'] = enc_submitted_via
    
    # Encoding for Company response to consumer
    company_response = df['Company response to consumer'].values[0]
    enc_company_response = encoded_values['Company response to consumer'][company_response]
    df['Company response to consumer'] = enc_company_response
    
    # Encoding for Timely response?
    timely_response = df['Timely response?'].values[0]
    enc_timely_response = encoded_values['Timely response?'][timely_response]
    df['Timely response?'] = enc_timely_response

    

    # Convert DataFrame to PyTorch tensor
    data = torch.tensor(df.values).float()
    
    with torch.no_grad():
        print("Input: ", data)
        output = model(data)
        print("Model output: ", output)
        prediction = torch.sigmoid(output)
        prediction = torch.round(prediction).numpy().tolist()
    
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)

