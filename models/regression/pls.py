from sklearn.cross_decomposition import PLSRegression
import numpy as np
import pandas as pd
import streamlit as st
from ..base import BaseModel

class PLSRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'n_components': {
                'type': 'number_input',
                'label': 'Number of Components',
                'min_value': 1,
                'max_value': 20,  # Reasonable upper limit
                'value': 2,
                'step': 1,
                'help': "Number of PLS components to extract. Should be less than or equal to the number of features."
            },
            'scale': {
                'type': 'checkbox',
                'label': 'Scale Features',
                'value': True,
                'help': "Whether to standardize (zero-mean, unit-variance) each feature before fitting"
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 100,
                'max_value': 1000,
                'value': 500,
                'step': 100,
                'help': "Maximum number of iterations for the NIPALS algorithm"
            },
            'tol': {
                'type': 'number_input',
                'label': 'Convergence Tolerance',
                'min_value': 1e-8,
                'max_value': 1e-3,
                'value': 1e-6,
                'step': 1e-6,
                'help': "Convergence tolerance for the iterative NIPALS updates"
            },
            'copy': {
                'type': 'checkbox',
                'label': 'Copy Data',
                'value': True,
                'help': "If True, the input data X and Y are copied before fitting; if False, they may be overwritten in place"
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        self.model = PLSRegression(**kwargs)
        self.model.fit(X, y)
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 