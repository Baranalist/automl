from sklearn.linear_model import TheilSenRegressor
import streamlit as st
from ..base import BaseModel

class TheilSenModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = TheilSenRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to calculate the intercept for this model."
            },
            'copy_X': {
                'type': 'checkbox',
                'label': 'Copy X',
                'value': True,
                'help': "If True, X will be copied; else, it may be overwritten."
            },
            'max_subpopulation': {
                'type': 'number_input',
                'label': 'Max Subpopulation',
                'min_value': 1.0,
                'value': 10000.0,
                'step': 1000.0,
                'help': "Max number of combinations of samples from which slopes are computed. Higher values give more accurate results but are slower."
            },
            'n_subsamples': {
                'type': 'number_input',
                'label': 'Number of Subsamples',
                'min_value': 1,
                'value': 30,
                'help': "Number of samples drawn to compute candidate slopes. Must be > number of features."
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 1,
                'value': 300,
                'help': "Maximum number of iterations for refining the median slope."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Tolerance',
                'min_value': 0.0,
                'value': 0.001,
                'step': 0.0001,
                'help': "Tolerance to declare convergence."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': 42,
                'help': "Controls randomness of subsampling."
            },
            'verbose': {
                'type': 'selectbox',
                'label': 'Verbosity Level',
                'options': ['0', '1', '2'],
                'help': "Controls amount of logging output. 0: no output, 1: basic output, 2: detailed output."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Convert verbose string to integer
        if 'verbose' in kwargs:
            kwargs['verbose'] = int(kwargs['verbose'])
        
        self.model = TheilSenRegressor(**kwargs)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 