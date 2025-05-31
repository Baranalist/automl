from sklearn.linear_model import Ridge
import streamlit as st
from ..base import BaseModel

class RidgeRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = Ridge()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'alpha': {
                'type': 'slider',
                'label': 'Regularization Strength (alpha)',
                'min_value': 0.0,
                'max_value': 100.0,
                'value': 1.0,
                'step': 0.01,
                'help': "Regularization strength. Higher values increase regularization."
            },
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
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 1,
                'value': 1000,
                'help': "Maximum number of iterations for solvers 'sag', 'lsqr', or 'saga'."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Tolerance',
                'min_value': 0.0,
                'value': 0.001,
                'step': 0.0001,
                'help': "Precision tolerance for solver convergence."
            },
            'solver': {
                'type': 'selectbox',
                'label': 'Solver',
                'options': ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
                'help': "Algorithm used to solve the ridge regression."
            },
            'positive': {
                'type': 'checkbox',
                'label': 'Positive Coefficients',
                'value': False,
                'help': "If True, restricts coefficients to be ≥ 0. Only supported by solver='saga'."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': 42,
                'help': "Seed for reproducibility. Only used for solver='sag' or 'saga'."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        self.model = Ridge(**kwargs)
        self.model.fit(X, y)
        return self.model 