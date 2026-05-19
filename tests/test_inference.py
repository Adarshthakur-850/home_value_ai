import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predict import HousePricePredictor

def test_inference():
    print("Testing inference...")
    try:
        predictor = HousePricePredictor()
        input_data = {
            'sqft': 2500,
            'bedrooms': 4,
            'bathrooms': 3,
            'floors': 2,
            'zipcode': '98004',
            'year_built': 2010,
            'condition': 5
        }
        
        price = predictor.predict(input_data)
        print(f"Predicted Price: ${price:,.2f}")
        
        assert price > 0, "Price should be positive"
        print("Inference test passed!")
    except Exception as e:
        print(f"Inference test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_inference()
