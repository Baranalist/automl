from sklearn.linear_model import LinearRegression
import streamlit as st
from ..base import BaseModel

class LinearRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
    
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
            'n_jobs': {
                'type': 'number_input',
                'label': 'Number of Jobs',
                'min_value': -1,
                'value': None,
                'help': "The number of jobs to use for the computation. -1 means using all processors."
            },
            'positive': {
                'type': 'checkbox',
                'label': 'Positive Coefficients',
                'value': False,
                'help': "When set to True, forces the coefficients to be positive."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        self.model = LinearRegression(**kwargs)
        self.model.fit(X, y)
        return self.model 

    def predict(self, X):
        """Make predictions using the trained model"""
        return self.model.predict(X) 