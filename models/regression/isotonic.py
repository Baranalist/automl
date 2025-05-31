from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd
import streamlit as st
from ..base import BaseModel

class IsotonicRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'y_min': {
                'type': 'number_input',
                'label': 'Minimum Y Value',
                'min_value': None,
                'max_value': None,
                'value': None,
                'step': 0.1,
                'help': "Lower bound on fitted predictions. Leave empty for no lower bound."
            },
            'y_max': {
                'type': 'number_input',
                'label': 'Maximum Y Value',
                'min_value': None,
                'max_value': None,
                'value': None,
                'step': 0.1,
                'help': "Upper bound on fitted predictions. Leave empty for no upper bound."
            },
            'increasing': {
                'type': 'selectbox',
                'label': 'Monotonicity Direction',
                'options': ['increasing', 'decreasing'],
                'help': "Whether the fitted function should be non-decreasing (increasing) or non-increasing (decreasing)"
            },
            'out_of_bounds': {
                'type': 'selectbox',
                'label': 'Out of Bounds Behavior',
                'options': ['nan', 'clip', 'raise'],
                'help': "How to handle predictions outside the training range: 'nan' returns NaN, 'clip' clips to boundaries, 'raise' raises an error"
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Get parameters
        y_min = kwargs.get('y_min')
        y_max = kwargs.get('y_max')
        increasing = kwargs.get('increasing', 'increasing') == 'increasing'
        out_of_bounds = kwargs.get('out_of_bounds', 'nan')
        
        # Create and train model
        self.model = IsotonicRegression(
            y_min=y_min,
            y_max=y_max,
            increasing=increasing,
            out_of_bounds=out_of_bounds
        )
        
        # For isotonic regression, we need to sort X and y
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].values  # Use first column for sorting and convert to numpy array
        else:
            X = np.array(X)
        
        if isinstance(y, pd.Series):
            y = y.values  # Convert to numpy array
        
        # Sort data by X values
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]
        
        self.model.fit(X_sorted, y_sorted)
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # For isotonic regression, we need to sort X
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].values  # Use first column for sorting and convert to numpy array
        else:
            X = np.array(X)
        
        # Sort X values
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        
        # Make predictions
        y_pred = self.model.predict(X_sorted)
        
        # Restore original order
        y_pred_original = np.zeros_like(y_pred)
        y_pred_original[sort_idx] = y_pred
        
        return y_pred_original 