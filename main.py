import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import joblib
from firebase_config import get_branch_wait_times
from train_model import WaitTimePredictor as TrainPredictor

app = Flask(__name__)

class WaitTimePredictor:
    def __init__(self):
        self.model = None
        self.scaler_y = None
        self.sequence_length = 5
        self.model_path = 'wait_time_model'
        self.scaler_path = 'wait_time_scaler.pkl'
        self.initialize_model()
        
    def initialize_model(self):
        """Initialize the model, training if necessary"""
        model_file = os.path.join(self.model_path, 'model.h5')
        scaler_file = os.path.join(self.model_path, self.scaler_path)
        
        if not os.path.exists(model_file) or not os.path.exists(scaler_file):
            print("Model or scaler not found. Training new model...")
            trainer = TrainPredictor()
            test_rmse = trainer.train('dataset/Restaurant.csv')
            print(f"Model trained successfully with test RMSE: {test_rmse:.2f} minutes")
        
        self.load_model()
        
    def load_model(self):
        """Load the trained model and scaler"""
        model_file = os.path.join(self.model_path, 'model.h5')
        scaler_file = os.path.join(self.model_path, self.scaler_path)
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            self.model = load_model(model_file)
            self.scaler_y = joblib.load(scaler_file)
            return True
        return False

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

# Initialize predictor
predictor = WaitTimePredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        branch_id = data.get('branch_id')
        
        if not branch_id:
            return jsonify({
                'status': 'error',
                'message': 'branch_id is required'
            }), 400
        
        try:
            # Get recent wait times for the branch
            recent_waits = get_branch_wait_times(branch_id, limit=5)
            
            if len(recent_waits) != predictor.sequence_length:
                return jsonify({
                    'status': 'error',
                    'message': f'Not enough historical data available for prediction'
                }), 400
                
            prediction = predictor.predict_wait_time(recent_waits)
            
            # Calculate how many times are real vs synthetic
            real_count = len([t for t in recent_waits if isinstance(t, (int, float)) and t > 0])
            synthetic_count = len(recent_waits) - real_count
            
            print(recent_waits)
            
            return jsonify({
                'status': 'success',
                'predicted_wait_time': float(prediction),
                'recent_waits': recent_waits,
                'data_quality': {
                    'real_data_count': real_count,
                    'synthetic_data_count': synthetic_count,
                    'is_partially_synthetic': synthetic_count > 0,
                    'is_fully_synthetic': synthetic_count == len(recent_waits)
                }
            })
            
        except ValueError as ve:
            # Handle invalid branch ID
            return jsonify({
                'status': 'error',
                'message': str(ve)
            }), 404
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predict-test', methods=['POST']) 
def predict_from_times():
    try:
        data = request.json
        wait_times = data.get('wait_times', [])
        
        if not isinstance(wait_times, list):
            return jsonify({
                'status': 'error',
                'message': 'wait_times must be a list of numbers'
            }), 400
            
        if len(wait_times) != predictor.sequence_length:
            return jsonify({
                'status': 'error',
                'message': f'Please provide exactly {predictor.sequence_length} wait times'
            }), 400
            
        # Validate all inputs are numbers
        try:
            wait_times = [float(t) for t in wait_times]
        except (ValueError, TypeError):
            return jsonify({
                'status': 'error',
                'message': 'All wait times must be valid numbers'
            }), 400
            
        prediction = predictor.predict_wait_time(wait_times)
        
        return jsonify({
            'status': 'success',
            'input_wait_times': wait_times,
            'predicted_wait_time': float(prediction)
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')