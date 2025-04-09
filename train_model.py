import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging (1=INFO, 2=WARNING, 3=ERROR)

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib

# Create required directories if they don't exist
# IMAGE_PATH = 'output'
# DATASET_PATH = 'dataset'
# for path in [IMAGE_PATH, DATASET_PATH]:
#     if not os.path.exists(path):
#         os.makedirs(path)

class WaitTimePredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.sequence_length = 5
        self.model_path = 'wait_time_model'
        self.scaler_path = 'wait_time_scaler.pkl'
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
    def create_sequences(self, data):
        """Convert data into sequences for LSTM"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def create_time_features(self, df):
        """Create time-based features"""
        base_date = datetime.now() - timedelta(days=7)
        timestamps = [base_date + timedelta(minutes=x*60) for x in range(len(df))]
        
        df['timestamp'] = timestamps
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        
        return df
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(32, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'  # Changed from 'mse' to 'mean_squared_error'
        )
        
        return model
    
    def analyze_data(self, df):
        """Analyze the dataset and print insights"""
        print("\nData Analysis:")
        print(f"Number of samples: {len(df)}")
        
        wait_times = df['Wait Time (in minutes)']
        print("\nWait Time Statistics:")
        print(f"Mean: {wait_times.mean():.2f} minutes")
        print(f"Median: {wait_times.median():.2f} minutes")
        print(f"Std Dev: {wait_times.std():.2f} minutes")
        print(f"Min: {wait_times.min():.2f} minutes")
        print(f"Max: {wait_times.max():.2f} minutes")
        
        # plt.figure(figsize=(15, 5))
        # plt.plot(df['timestamp'], df['Wait Time (in minutes)'])
        # plt.title('Wait Times Over Time')
        # plt.xlabel('Time')
        # plt.ylabel('Wait Time (minutes)')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig(f'{IMAGE_PATH}/wait_time_series.png')
        # plt.close()

    def save_model(self):
        """Save the trained model and scaler"""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model.save(os.path.join(self.model_path, 'model.h5'))
        joblib.dump(self.scaler_y, os.path.join(self.model_path, self.scaler_path))

    def train(self, data_path):
        # Load data
        df = pd.read_csv(data_path)
        
        # Create time features
        df = self.create_time_features(df)
        
        # Analyze data
        self.analyze_data(df)
        
        # Prepare data for sequence prediction
        wait_times = df['Wait Time (in minutes)'].values.reshape(-1, 1)
        
        # Scale the data
        wait_times_scaled = self.scaler_y.fit_transform(wait_times)
        
        # Create sequences
        X, y = self.create_sequences(wait_times_scaled)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split into train and test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            verbose=1
        )
        
        # Save the trained model and scaler
        self.save_model()
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Inverse transform predictions
        train_pred = self.scaler_y.inverse_transform(train_pred)
        test_pred = self.scaler_y.inverse_transform(test_pred)
        y_train_actual = self.scaler_y.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate metrics
        train_rmse = np.sqrt(np.mean((y_train_actual - train_pred) ** 2))
        test_rmse = np.sqrt(np.mean((y_test_actual - test_pred) ** 2))
        
        print("\nModel Performance:")
        print(f"Train RMSE: {train_rmse:.2f} minutes")
        print(f"Test RMSE: {test_rmse:.2f} minutes")
        
        # Plot predictions vs actual
        # plt.figure(figsize=(15, 5))
        # plt.plot(y_test_actual, label='Actual')
        # plt.plot(test_pred, label='Predicted')
        # plt.title('Actual vs Predicted Wait Times')
        # plt.xlabel('Time Step')
        # plt.ylabel('Wait Time (minutes)')
        # plt.legend()
        # plt.savefig(f'{IMAGE_PATH}/predictions.png')
        # plt.close()
        
        # Plot training history
        # plt.figure(figsize=(10, 5))
        # plt.plot(history.history['loss'], label='Training Loss')
        # plt.plot(history.history['val_loss'], label='Validation Loss')
        # plt.title('Model Loss During Training')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.savefig(f'{IMAGE_PATH}/training_history.png')
        # plt.close()
        
        return test_rmse

    def predict_wait_time(self, recent_waits):
        """Predict next wait time based on recent wait times"""
        if self.model is None or self.scaler_y is None:
            raise ValueError("Model not loaded. Please ensure the model is trained first.")
            
        recent_waits = np.array(recent_waits).reshape(-1, 1)
        recent_waits_scaled = self.scaler_y.transform(recent_waits)
        sequence = recent_waits_scaled.reshape(1, self.sequence_length, 1)
        prediction_scaled = self.model.predict(sequence)
        prediction = self.scaler_y.inverse_transform(prediction_scaled)[0][0]
        return max(0, prediction)

if __name__ == "__main__":
    predictor = WaitTimePredictor()
    test_rmse = predictor.train('dataset/Restaurant.csv')
    
    predict = predictor.predict_wait_time([2.0647499998410543, 1, 1.007264088275515, 1, 3.04752326342894])
    
    print(f"Predicted wait time: {predict:.2f} minutes")
    print(f"\nTraining complete! Model saved with test RMSE: {test_rmse:.2f} minutes")