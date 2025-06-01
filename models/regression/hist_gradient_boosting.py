from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
import streamlit as st
from ..base import BaseModel

class HistGradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = HistGradientBoostingRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'loss': {
                'type': 'selectbox',
                'label': 'Loss Function',
                'options': ['squared_error', 'absolute_error', 'poisson', 'quantile', 'huber'],
                'help': "Loss function to be minimized. 'squared_error' for mean squared error, 'absolute_error' for mean absolute error, 'poisson' for Poisson regression, 'quantile' for quantile regression, 'huber' for Huber loss."
            },
            'learning_rate': {
                'type': 'number_input',
                'label': 'Learning Rate',
                'min_value': 0.001,
                'max_value': 1.0,
                'value': 0.1,
                'step': 0.01,
                'help': "Shrinks the contribution of each tree. Lower values require more trees but can lead to better generalization."
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 1,
                'max_value': 10000,
                'value': 100,
                'step': 10,
                'help': "Number of boosting iterations (trees) to perform. More trees generally give better performance but take longer to train."
            },
            'max_leaf_nodes': {
                'type': 'number_input',
                'label': 'Maximum Leaf Nodes',
                'min_value': 2,
                'max_value': 255,
                'value': 31,
                'step': 1,
                'help': "Maximum number of leaf nodes per tree. Higher values make the model more complex and more likely to overfit."
            },
            'max_depth': {
                'type': 'number_input',
                'label': 'Maximum Tree Depth',
                'min_value': 1,
                'max_value': 100,
                'value': None,
                'step': 1,
                'help': "Maximum depth of each tree. None means unlimited depth. Higher values make the model more complex and more likely to overfit."
            },
            'min_samples_leaf': {
                'type': 'number_input',
                'label': 'Minimum Samples per Leaf',
                'min_value': 1,
                'max_value': 1000,
                'value': 20,
                'step': 1,
                'help': "Minimum number of samples required to form a leaf node. Higher values make the model more conservative."
            },
            'l2_regularization': {
                'type': 'number_input',
                'label': 'L2 Regularization',
                'min_value': 0.0,
                'max_value': 100.0,
                'value': 0.0,
                'step': 0.1,
                'help': "L2 regularization strength. Higher values make the model more conservative."
            },
            'max_bins': {
                'type': 'number_input',
                'label': 'Maximum Bins',
                'min_value': 2,
                'max_value': 1024,
                'value': 255,
                'step': 1,
                'help': "Maximum number of bins to bucket continuous features. Higher values can lead to better precision but slower training."
            },
            'early_stopping': {
                'type': 'checkbox',
                'label': 'Early Stopping',
                'value': False,
                'help': "Whether to use early stopping based on validation data. If True, will stop training if validation score doesn't improve."
            },
            'validation_fraction': {
                'type': 'number_input',
                'label': 'Validation Fraction',
                'min_value': 0.1,
                'max_value': 0.5,
                'value': 0.1,
                'step': 0.05,
                'help': "Proportion of training data to use for validation if early_stopping=True."
            },
            'n_iter_no_change': {
                'type': 'number_input',
                'label': 'Iterations Without Change',
                'min_value': 1,
                'max_value': 100,
                'value': 10,
                'step': 1,
                'help': "Number of iterations without improvement before stopping if early_stopping=True."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Tolerance',
                'min_value': 0.00001,
                'max_value': 0.1,
                'value': 0.0001,
                'step': 0.0001,
                'help': "Minimum change in validation score to qualify as an improvement for early stopping."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random Seed',
                'min_value': 0,
                'max_value': 1000,
                'value': 42,
                'step': 1,
                'help': "Random number seed for reproducibility."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Create a new dictionary for HistGradientBoosting parameters
        hist_params = {}
        
        # Add all parameters that are valid for HistGradientBoostingRegressor
        valid_params = [
            'loss', 'learning_rate', 'max_iter', 'max_leaf_nodes',
            'max_depth', 'min_samples_leaf', 'l2_regularization',
            'max_bins', 'early_stopping', 'validation_fraction',
            'n_iter_no_change', 'tol', 'random_state'
        ]
        
        # Handle special cases
        if kwargs.get('max_depth') == 0:  # Convert 0 to None for unlimited depth
            kwargs['max_depth'] = None
        
        # Add parameters
        for param in valid_params:
            if param in kwargs:
                hist_params[param] = kwargs[param]
        
        self.model = HistGradientBoostingRegressor(**hist_params)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 