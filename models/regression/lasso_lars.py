from sklearn.linear_model import LassoLars
import numpy as np
import pandas as pd
from ..base import BaseModel

class LassoLARSModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'alpha': {
                'type': 'number_input',
                'label': 'Regularization Parameter',
                'min_value': 0.0,
                'max_value': 10.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Regularization parameter. If alpha = 0, equivalent to ordinary LARS; larger values yield stronger L1 penalty"
            },
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to include an intercept term"
            },
            'verbose': {
                'type': 'number_input',
                'label': 'Verbosity Level',
                'min_value': 0,
                'max_value': 10,
                'value': 0,
                'step': 1,
                'help': "Controls verbosity during fitting"
            },
            'precompute': {
                'type': 'selectbox',
                'label': 'Precompute',
                'options': ['auto', 'True', 'False'],
                'value': 'auto',
                'help': "Whether to use a precomputed Gram matrix to speed up the computation"
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 100,
                'max_value': 1000,
                'value': 500,
                'step': 100,
                'help': "Maximum number of iterations for coordinate descent when refining the solution path"
            },
            'eps': {
                'type': 'number_input',
                'label': 'Numerical Stability Threshold',
                'min_value': 1e-10,
                'max_value': 1e-2,
                'value': 1e-6,
                'step': 1e-6,
                'help': "Tiny constant added for numerical stability"
            },
            'copy_X': {
                'type': 'checkbox',
                'label': 'Copy X',
                'value': True,
                'help': "If True, X is copied before fitting; if False, X may be overwritten"
            },
            'fit_path': {
                'type': 'checkbox',
                'label': 'Fit Path',
                'value': True,
                'help': "If True, store the entire LARS solution path (coefficients at each step); else only the final solution"
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        self.model = LassoLars(**kwargs)
        self.model.fit(X, y)
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 