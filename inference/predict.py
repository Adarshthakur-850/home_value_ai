import joblib
import pandas as pd
import os
import sys

# Add project root to path to ensure modules like preprocessing are importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import FeatureEngineer

class HousePricePredictor:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'price_model.pkl')
        self.model = None
        
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = joblib.load(self.model_path)
        
    def predict(self, input_data):
        """
        Args:
            input_data: dict or pd.DataFrame
        """
        if self.model is None:
            self.load_model()
            
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data
            
        prediction = self.model.predict(input_df)
        return prediction[0]

if __name__ == "__main__":
    predictor = HousePricePredictor()
    sample_input = {
        'sqft': 2000,
        'bedrooms': 3,
        'bathrooms': 2,
        'floors': 1,
        'zipcode': '98001',
        'year_built': 2000,
        'condition': 3
    }
    print("Predicted Price:", predictor.predict(sample_input))
