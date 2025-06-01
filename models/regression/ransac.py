from sklearn.linear_model import RANSACRegressor, LinearRegression
import streamlit as st
from ..base import BaseModel

class RANSACModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RANSACRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'min_samples': {
                'type': 'selectbox',
                'label': 'Minimum Samples Type',
                'options': ['fraction', 'absolute'],
                'help': "Choose whether to specify minimum samples as a fraction of total samples or as an absolute number."
            },
            'min_samples_fraction': {
                'type': 'slider',
                'label': 'Minimum Samples (Fraction)',
                'min_value': 0.1,
                'max_value': 1.0,
                'value': 0.5,
                'step': 0.1,
                'help': "Fraction of total samples to fit the base estimator."
            },
            'min_samples_absolute': {
                'type': 'number_input',
                'label': 'Minimum Samples (Absolute)',
                'min_value': 1,
                'value': 10,
                'help': "Absolute number of samples to fit the base estimator."
            },
            'residual_threshold': {
                'type': 'number_input',
                'label': 'Residual Threshold',
                'min_value': 0.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Maximum residual for a data point to be classified as an inlier. If None, estimated using MAD."
            },
            'max_trials': {
                'type': 'number_input',
                'label': 'Maximum Trials',
                'min_value': 1,
                'value': 100,
                'help': "Maximum number of iterations to perform."
            },
            'max_skips': {
                'type': 'number_input',
                'label': 'Maximum Skips',
                'min_value': 1,
                'value': 100,
                'help': "Maximum number of iterations that can be skipped due to invalid data or model."
            },
            'stop_n_inliers': {
                'type': 'number_input',
                'label': 'Stop N Inliers',
                'min_value': 0,
                'value': 0,
                'help': "If this number of inliers is reached, RANSAC stops early. Set to 0 to disable."
            },
            'stop_score': {
                'type': 'number_input',
                'label': 'Stop Score',
                'min_value': 0.0,
                'value': 0.0,
                'help': "If this score is reached by inlier data, RANSAC stops early. Set to 0 to disable."
            },
            'stop_probability': {
                'type': 'slider',
                'label': 'Stop Probability',
                'min_value': 0.0,
                'max_value': 1.0,
                'value': 0.99,
                'step': 0.01,
                'help': "Desired probability that at least one valid set of inliers is sampled."
            },
            'loss': {
                'type': 'selectbox',
                'label': 'Loss Function',
                'options': ['absolute_error'],
                'help': "Loss function to compute residuals. Currently only absolute_error is supported."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': 42,
                'help': "Controls randomness of subsampling. Useful for reproducibility."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Handle min_samples parameter
        if 'min_samples' in kwargs:
            min_samples_type = kwargs.pop('min_samples')
            if min_samples_type == 'fraction':
                min_samples = kwargs.pop('min_samples_fraction', 0.5)
            else:  # absolute
                min_samples = kwargs.pop('min_samples_absolute', 10)
            kwargs['min_samples'] = min_samples
        
        # Remove any unexpected parameters
        expected_params = {
            'min_samples', 'residual_threshold', 'max_trials', 'max_skips',
            'stop_n_inliers', 'stop_score', 'stop_probability', 'loss',
            'random_state'
        }
        kwargs = {k: v for k, v in kwargs.items() if k in expected_params}
        
        self.model = RANSACRegressor(**kwargs)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 