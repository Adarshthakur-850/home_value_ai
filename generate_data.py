import pandas as pd
import numpy as np
import os

def generate_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'sqft': np.random.randint(500, 5000, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 5, n_samples),
        'floors': np.random.randint(1, 4, n_samples),
        'zipcode': np.random.choice(['98001', '98002', '98003', '98004', '98005'], n_samples),
        'year_built': np.random.randint(1950, 2024, n_samples),
        'condition': np.random.randint(1, 6, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate price based on features (simple linear relation + noise)
    base_price = 50000
    price_per_sqft = 200
    
    df['price'] = base_price + \
                  (df['sqft'] * price_per_sqft) + \
                  (df['bedrooms'] * 10000) + \
                  (df['bathrooms'] * 15000) + \
                  (df['year_built'] - 1950) * 1000 + \
                  (df['condition'] * 5000) + \
                  np.random.normal(0, 20000, n_samples)
                  
    # Add some categorical influence (Zipcode)
    zipcode_multiplier = {'98001': 1.0, '98002': 1.1, '98003': 1.2, '98004': 1.5, '98005': 1.3}
    df['price'] = df.apply(lambda row: row['price'] * zipcode_multiplier[row['zipcode']], axis=1)
    
    return df

if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    df = generate_data()
    df.to_csv('data/housing.csv', index=False)
    print("Dummy data generated at data/housing.csv")
