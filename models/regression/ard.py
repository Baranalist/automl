from sklearn.linear_model import ARDRegression
import numpy as np
import streamlit as st
from ..base import BaseModel

class ARDRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'n_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 100,
                'max_value': 1000,
                'value': 300,
                'step': 100,
                'help': "Maximum number of iterations in the evidence maximization loop"
            },
            'tol': {
                'type': 'number_input',
                'label': 'Convergence Tolerance',
                'min_value': 1e-6,
                'max_value': 1e-2,
                'value': 1e-3,
                'step': 1e-4,
                'help': "Convergence threshold; if changes in weight estimates fall below this, training stops"
            },
            'alpha_1': {
                'type': 'number_input',
                'label': 'Alpha 1 (Noise Precision Shape)',
                'min_value': 1e-6,
                'max_value': 1e-4,
                'value': 1e-6,
                'step': 1e-6,
                'help': "Shape parameter for the Gamma prior over noise precision. Lower values mean a broader prior"
            },
            'alpha_2': {
                'type': 'number_input',
                'label': 'Alpha 2 (Noise Precision Rate)',
                'min_value': 1e-6,
                'max_value': 1e-4,
                'value': 1e-6,
                'step': 1e-6,
                'help': "Rate parameter for the Gamma prior over noise precision. Controls noise penalty"
            },
            'lambda_1': {
                'type': 'number_input',
                'label': 'Lambda 1 (Weight Precision Shape)',
                'min_value': 1e-6,
                'max_value': 1e-4,
                'value': 1e-6,
                'step': 1e-6,
                'help': "Shape parameter for the Gamma prior over weight precision. Lower values mean a broader prior"
            },
            'lambda_2': {
                'type': 'number_input',
                'label': 'Lambda 2 (Weight Precision Rate)',
                'min_value': 1e-6,
                'max_value': 1e-4,
                'value': 1e-6,
                'step': 1e-6,
                'help': "Rate parameter for the Gamma prior over weight precision. Controls weight penalty"
            },
            'compute_score': {
                'type': 'checkbox',
                'label': 'Compute Evidence Score',
                'value': False,
                'help': "Whether to compute the marginal log-likelihood at each iteration (slower but provides more information)"
            },
            'threshold_lambda': {
                'type': 'number_input',
                'label': 'Feature Pruning Threshold',
                'min_value': 1e3,
                'max_value': 1e6,
                'value': 1e4,
                'step': 1e3,
                'help': "Features with weight precision above this threshold are pruned (coefficient set to zero)"
            },
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to calculate the intercept for this model"
            },
            'copy_X': {
                'type': 'checkbox',
                'label': 'Copy X',
                'value': True,
                'help': "If False, X may be overwritten in-place to save memory"
            },
            'verbose': {
                'type': 'checkbox',
                'label': 'Verbose Output',
                'value': False,
                'help': "Whether to print progress messages during fitting"
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Get parameters
        n_iter = kwargs.get('n_iter', 300)
        tol = kwargs.get('tol', 1e-3)
        alpha_1 = kwargs.get('alpha_1', 1e-6)
        alpha_2 = kwargs.get('alpha_2', 1e-6)
        lambda_1 = kwargs.get('lambda_1', 1e-6)
        lambda_2 = kwargs.get('lambda_2', 1e-6)
        compute_score = kwargs.get('compute_score', False)
        threshold_lambda = kwargs.get('threshold_lambda', 1e4)
        fit_intercept = kwargs.get('fit_intercept', True)
        copy_X = kwargs.get('copy_X', True)
        verbose = kwargs.get('verbose', False)
        
        # Create and train model
        self.model = ARDRegression(
            n_iter=n_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            compute_score=compute_score,
            threshold_lambda=threshold_lambda,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose
        )
        
        self.model.fit(X, y)
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X)