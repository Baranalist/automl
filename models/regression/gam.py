from pygam import LinearGAM
import numpy as np
import pandas as pd
from ..base import BaseModel

class GAMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'n_splines': {
                'type': 'number_input',
                'label': 'Number of Splines',
                'min_value': 10,
                'max_value': 50,
                'value': 25,
                'step': 1,
                'help': "Number of basis functions for spline terms. Higher values allow more complex relationships but may lead to overfitting."
            },
            'spline_order': {
                'type': 'number_input',
                'label': 'Spline Order',
                'min_value': 1,
                'max_value': 5,
                'value': 3,
                'step': 1,
                'help': "Order of spline (degree + 1). 3 is cubic, 4 is quartic, etc. Higher orders allow more complex relationships."
            },
            'lam': {
                'type': 'number_input',
                'label': 'Smoothing Parameter',
                'min_value': 0.0001,
                'max_value': 10.0,
                'value': 0.6,
                'step': 0.1,
                'help': "Smoothing parameter that controls the trade-off between fit and smoothness. Higher values result in smoother fits."
            },
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to include a constant (bias) term in the model."
            },
            'fit_linear': {
                'type': 'checkbox',
                'label': 'Fit Linear Terms',
                'value': False,
                'help': "Whether to include unpenalized linear terms for each feature."
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 50,
                'max_value': 1000,
                'value': 100,
                'step': 50,
                'help': "Maximum number of iterations for the solver to converge."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Convergence Tolerance',
                'min_value': 1e-6,
                'max_value': 1e-2,
                'value': 1e-4,
                'step': 1e-4,
                'help': "Tolerance for stopping criteria. If the change in objective is below this value, fitting stops."
            },
            'verbose': {
                'type': 'checkbox',
                'label': 'Verbose Output',
                'value': False,
                'help': "If True, print out iteration logs and warnings during fitting."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Get parameters
        n_splines = kwargs.get('n_splines', 25)
        spline_order = kwargs.get('spline_order', 3)
        lam = kwargs.get('lam', 0.6)
        fit_intercept = kwargs.get('fit_intercept', True)
        fit_linear = kwargs.get('fit_linear', False)
        max_iter = kwargs.get('max_iter', 100)
        tol = kwargs.get('tol', 1e-4)
        verbose = kwargs.get('verbose', False)
        
        # Create and train model
        self.model = LinearGAM(
            n_splines=n_splines,
            spline_order=spline_order,
            lam=lam,
            fit_intercept=fit_intercept,
            fit_linear=fit_linear,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose
        )
        
        self.model.fit(X, y)
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 