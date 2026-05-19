import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import get_preprocessor, FeatureEngineer

def train_model():
    print("Loading data...")
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'housing.csv')
    df = pd.read_csv(data_path)
    
    # Feature Engineering (Age calculation happening inside pipeline but needed for column transformer setup if not careful)
    # Actually, the FeatureEngineer transformer adds 'age', so we need to make sure the preprocessor knows about it.
    # The preprocessor expects 'age' to be present.
    
    X = df.drop('price', axis=1)
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = -float('inf')
    best_name = ""
    
    print("Training models...")
    for name, model in models.items():
        # Create full pipeline: Feature Eng -> Preprocess -> Model
        pipeline = Pipeline(steps=[
            ('feature_eng', FeatureEngineer()),
            ('preprocessor', get_preprocessor()),
            ('regressor', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n{name} Results:")
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = pipeline
            best_name = name
            
    print(f"\nBest Model: {best_name} with R2: {best_score:.4f}")
    
    # Save best model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'price_model.pkl')
    
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
