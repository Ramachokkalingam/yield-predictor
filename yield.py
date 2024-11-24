from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import os

app = Flask(__name__)

# Function to create sequences for LSTM
def create_sequences(features, target, time_steps):
    X, y = [], []
    for i in range(len(features) - time_steps):
        X.append(features[i:i + time_steps])
        y.append(target[i + time_steps])
    return np.array(X), np.array(y)

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')  # An HTML file for user interaction

# Route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    # Check if the uploaded file is an Excel file
    if not file.filename.endswith('.xlsx'):
        return jsonify({'error': 'Invalid file type. Please upload an Excel file.'}), 400

    # Load the Excel file
    data = pd.read_excel(file, sheet_name=0)

    # Process the data
    try:
        features = data.drop(columns=['Year', 'Yield (kg/hectare)']).values
        target = data['Yield (kg/hectare)'].values

        # Scale features and target
        scaler_features = MinMaxScaler()
        scaler_target = MinMaxScaler()

        features_scaled = scaler_features.fit_transform(features)
        target_scaled = scaler_target.fit_transform(target.reshape(-1, 1))

        # Create sequences
        time_steps = 3
        X, y = create_sequences(features_scaled, target_scaled, time_steps)

        # Split data into train and test sets
        X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
        y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

        # Build the LSTM model
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(time_steps, X.shape[2])),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

        # Make predictions
        predicted_yield_scaled = model.predict(X_test)
        predicted_yield = scaler_target.inverse_transform(predicted_yield_scaled)
        actual_yield = scaler_target.inverse_transform(y_test)

        # Create a response with the first 5 predictions
        response = [
            {
                'Predicted Yield': float(predicted_yield[i][0]),
                'Actual Yield': float(actual_yield[i][0])
            } for i in range(min(len(predicted_yield), 5))
        ]

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
