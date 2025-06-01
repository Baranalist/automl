from sklearn.linear_model import QuantileRegressor
import streamlit as st
from ..base import BaseModel

class QuantileRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = QuantileRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'quantile': {
                'type': 'slider',
                'label': 'Quantile',
                'min_value': 0.01,
                'max_value': 0.99,
                'value': 0.5,
                'step': 0.01,
                'help': "The quantile to predict. 0.5 is the median regression."
            },
            'alpha': {
                'type': 'number_input',
                'label': 'Regularization Strength (alpha)',
                'min_value': 0.0,
                'value': 1.0,
                'step': 0.1,
                'help': "L2 regularization strength. Higher values mean stronger regularization. Set to 0 to disable regularization."
            },
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to calculate the intercept for this model."
            },
            'solver': {
                'type': 'selectbox',
                'label': 'Solver',
                'options': ['highs', 'highs-ds', 'highs-ipm', 'interior-point', 'revised simplex', 'simplex'],
                'help': "Solver used for underlying linear programming problem."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Solver Tolerance',
                'min_value': 0.0,
                'value': 1e-4,
                'step': 1e-4,
                'help': "Tolerance for the solver to converge (passed through solver_options)."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Handle tol parameter through solver_options
        if 'tol' in kwargs:
            tol = kwargs.pop('tol')
            kwargs['solver_options'] = {'tol': tol}
        
        self.model = QuantileRegressor(**kwargs)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 