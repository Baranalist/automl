from sklearn.linear_model import Ridge
import streamlit as st
from ..base import BaseModel

class RidgeRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'alpha': {
                'type': 'number_input',
                'label': 'Alpha',
                'min_value': 0.0,
                'value': 1.0,
                'help': "Regularization strength; must be a positive float."
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
                'label': 'Max Iterations',
                'min_value': 1,
                'value': None,
                'help': "Maximum number of iterations for conjugate gradient solver."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Tolerance',
                'min_value': 0.0,
                'value': 0.001,
                'help': "Precision of the solution."
            },
            'solver': {
                'type': 'selectbox',
                'label': 'Solver',
                'options': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'help': "Solver to use in the computational routines."
            },
            'positive': {
                'type': 'checkbox',
                'label': 'Positive Coefficients',
                'value': False,
                'help': "When set to True, forces the coefficients to be positive."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': None,
                'help': "Used when solver == 'sag' or 'saga' to shuffle the data."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        self.model = Ridge(**kwargs)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 