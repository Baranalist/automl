from sklearn.linear_model import Lasso
import streamlit as st
from ..base import BaseModel

class LassoRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = Lasso()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'alpha': {
                'type': 'slider',
                'label': 'Regularization Strength (alpha)',
                'min_value': 0.0,
                'max_value': 10.0,
                'value': 1.0,
                'step': 0.001,
                'help': "Regularization strength. Higher values increase regularization."
            },
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to calculate the intercept for this model."
            },
            'precompute': {
                'type': 'checkbox',
                'label': 'Precompute',
                'value': True,
                'help': "Whether to use a precomputed Gram matrix to speed up calculations."
            },
            'copy_X': {
                'type': 'checkbox',
                'label': 'Copy X',
                'value': True,
                'help': "If True, X will be copied; else, it may be overwritten."
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 1,
                'value': 1000,
                'help': "Maximum number of iterations for solver convergence."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Tolerance',
                'min_value': 0.0,
                'value': 0.001,
                'step': 0.0001,
                'help': "Tolerance for optimization convergence."
            },
            'warm_start': {
                'type': 'checkbox',
                'label': 'Warm Start',
                'value': False,
                'help': "Reuse solution of previous call to fit as initialization."
            },
            'positive': {
                'type': 'checkbox',
                'label': 'Positive Coefficients',
                'value': False,
                'help': "If True, forces coefficients to be ≥ 0."
            },
            'selection': {
                'type': 'selectbox',
                'label': 'Selection Strategy',
                'options': ['cyclic', 'random'],
                'help': "Coordinate descent selection strategy."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': 42,
                'help': "Seed for reproducibility. Only used when selection='random'."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        self.model = Lasso(**kwargs)
        self.model.fit(X, y)
        return self.model 