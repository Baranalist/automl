from sklearn.linear_model import HuberRegressor
import streamlit as st
from ..base import BaseModel

class HuberModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = HuberRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'epsilon': {
                'type': 'slider',
                'label': 'Epsilon',
                'min_value': 1.01,
                'max_value': 3.0,
                'value': 1.35,
                'step': 0.01,
                'help': "Controls the point where the loss function changes from quadratic to linear. Larger values make the model less sensitive to outliers."
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 1,
                'value': 100,
                'help': "Maximum number of iterations for optimization."
            },
            'alpha': {
                'type': 'slider',
                'label': 'Regularization Strength (alpha)',
                'min_value': 0.0,
                'max_value': 1.0,
                'value': 0.0001,
                'step': 0.0001,
                'help': "L2 regularization strength. Higher values increase regularization."
            },
            'warm_start': {
                'type': 'checkbox',
                'label': 'Warm Start',
                'value': False,
                'help': "Reuse solution of previous call to fit as initialization."
            },
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to calculate the intercept for this model."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Tolerance',
                'min_value': 0.0,
                'value': 0.001,
                'step': 0.0001,
                'help': "Tolerance for stopping criterion."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        self.model = HuberRegressor(**kwargs)
        self.model.fit(X, y)
        return self.model 