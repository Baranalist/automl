from sklearn.linear_model import BayesianRidge
import streamlit as st
from ..base import BaseModel

class BayesianRidgeModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = BayesianRidge()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'tol': {
                'type': 'number_input',
                'label': 'Tolerance',
                'min_value': 0.0,
                'value': 0.001,
                'step': 0.0001,
                'help': "Convergence threshold for optimization."
            },
            'alpha_1': {
                'type': 'number_input',
                'label': 'Alpha 1',
                'min_value': 1e-10,
                'value': 1e-6,
                'step': 1e-6,
                'format': '%.6f',
                'help': "Hyperparameter of the Gamma prior over the alpha parameter."
            },
            'alpha_2': {
                'type': 'number_input',
                'label': 'Alpha 2',
                'min_value': 1e-10,
                'value': 1e-6,
                'step': 1e-6,
                'format': '%.6f',
                'help': "Hyperparameter of the Gamma prior over the alpha parameter."
            },
            'lambda_1': {
                'type': 'number_input',
                'label': 'Lambda 1',
                'min_value': 1e-10,
                'value': 1e-6,
                'step': 1e-6,
                'format': '%.6f',
                'help': "Hyperparameter of the Gamma prior over the lambda parameter."
            },
            'lambda_2': {
                'type': 'number_input',
                'label': 'Lambda 2',
                'min_value': 1e-10,
                'value': 1e-6,
                'step': 1e-6,
                'format': '%.6f',
                'help': "Hyperparameter of the Gamma prior over the lambda parameter."
            },
            'compute_score': {
                'type': 'checkbox',
                'label': 'Compute Score',
                'value': False,
                'help': "Whether to compute the marginal log-likelihood at each iteration (slower)."
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
            'verbose': {
                'type': 'selectbox',
                'label': 'Verbosity Level',
                'options': ['False', 'True', '0', '1', '2'],
                'help': "Controls verbosity of the output. False/0: no output, True/1: basic output, 2: detailed output."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Convert verbose string to appropriate type
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
            if verbose == 'False':
                kwargs['verbose'] = False
            elif verbose == 'True':
                kwargs['verbose'] = True
            else:
                kwargs['verbose'] = int(verbose)
        
        self.model = BayesianRidge(**kwargs)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 