from sklearn.linear_model import Lars
import numpy as np
import pandas as pd
from ..base import BaseModel

class LARSRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to calculate and include an intercept term"
            },
            'verbose': {
                'type': 'number_input',
                'label': 'Verbosity Level',
                'min_value': 0,
                'max_value': 10,
                'value': 0,
                'step': 1,
                'help': "Level of verbosity during fit. 0 = silent, higher values print more details"
            },
            'precompute': {
                'type': 'selectbox',
                'label': 'Precompute',
                'options': ['auto', 'True', 'False'],
                'value': 'auto',
                'help': "Controls whether to use a precomputed Gram matrix to speed up computations"
            },
            'n_nonzero_coefs': {
                'type': 'number_input',
                'label': 'Number of Non-zero Coefficients',
                'min_value': 1,
                'max_value': 1000,  # Reasonable upper limit
                'value': 500,
                'step': 1,
                'help': "Target number of non-zero coefficients—stops when this many features have been entered"
            },
            'eps': {
                'type': 'number_input',
                'label': 'Numerical Stability Threshold',
                'min_value': 1e-10,
                'max_value': 1e-2,
                'value': 1e-6,
                'step': 1e-6,
                'help': "Threshold for numerical stability in the Cholesky decomposition"
            },
            'copy_X': {
                'type': 'checkbox',
                'label': 'Copy X',
                'value': True,
                'help': "If True, a copy of X is made before fitting. If False, X may be overwritten to save memory"
            },
            'fit_path': {
                'type': 'checkbox',
                'label': 'Fit Path',
                'value': True,
                'help': "If True, collect the entire solution path; else only store the final coefficients"
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        self.model = Lars(**kwargs)
        self.model.fit(X, y)
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 