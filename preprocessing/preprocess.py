import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        # Age of house
        current_year = 2024
        if 'year_built' in X.columns:
            X['age'] = current_year - X['year_built']
        return X

def get_preprocessor():
    """
    Returns a Scikit-Learn ColumnTransformer for preprocessing.
    """
    categorical_features = ['zipcode', 'condition']
    numeric_features = ['sqft', 'bedrooms', 'bathrooms', 'floors', 'age']
    
    # Validating features exist is done by ColumnTransformer during fit based on input df
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    return preprocessor

def preprocess_pipeline():
    """
    Returns full pipeline including feature engineering and preprocessing.
    """
    return Pipeline(steps=[
        ('feature_eng', FeatureEngineer()),
        ('preprocessor', get_preprocessor())
    ])
