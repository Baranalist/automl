from sklearn.linear_model import PoissonRegressor
import streamlit as st
from ..base import BaseModel

class PoissonRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = PoissonRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'alpha': {
                'type': 'number_input',
                'label': 'Regularization Strength (alpha)',
                'min_value': 0.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Regularization strength. Higher values mean stronger regularization. Set to 0 to disable regularization."
            },
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to calculate the intercept for this model."
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 1,
                'value': 100,
                'help': "Maximum number of iterations for the optimization algorithm."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Convergence Tolerance',
                'min_value': 0.0,
                'value': 1e-4,
                'step': 1e-4,
                'help': "The tolerance for the optimization algorithm to converge."
            },
            'warm_start': {
                'type': 'checkbox',
                'label': 'Warm Start',
                'value': False,
                'help': "When set to True, reuse the solution of the previous call to fit as initialization."
            },
            'verbose': {
                'type': 'selectbox',
                'label': 'Verbosity Level',
                'options': [0, 1, 2],
                'help': "Controls the amount of logging output. 0: no output, 1: some output, 2: detailed output."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Create a new dictionary for Poisson parameters
        poisson_params = {}
        
        # Add parameters that are valid for PoissonRegressor
        valid_params = [
            'alpha', 'fit_intercept', 'max_iter', 'tol',
            'warm_start', 'verbose'
        ]
        
        for param in valid_params:
            if param in kwargs:
                poisson_params[param] = kwargs[param]
        
        self.model = PoissonRegressor(**poisson_params)
        self.model.fit(X, y)
        return self.model 