from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel, ConstantKernel
import numpy as np
import streamlit as st
from ..base import BaseModel

class GaussianProcessRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'kernel_type': {
                'type': 'selectbox',
                'label': 'Kernel Type',
                'options': ['RBF', 'Matern', 'RationalQuadratic', 'ExpSineSquared', 'DotProduct', 'RBF + White', 'Constant * RBF'],
                'help': "Type of kernel function to use for the Gaussian Process"
            },
            'length_scale': {
                'type': 'number_input',
                'label': 'Length Scale',
                'min_value': 0.1,
                'max_value': 10.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Length scale parameter for the kernel function"
            },
            'alpha': {
                'type': 'number_input',
                'label': 'Alpha (Noise Level)',
                'min_value': 1e-10,
                'max_value': 1.0,
                'value': 1e-10,
                'step': 1e-10,
                'help': "Value added to the diagonal of the kernel matrix during fitting. Larger values mean more regularization/noise."
            },
            'optimizer': {
                'type': 'selectbox',
                'label': 'Optimizer',
                'options': ['fmin_l_bfgs_b', 'None'],
                'help': "Optimizer for kernel hyperparameters. 'fmin_l_bfgs_b' is recommended for most cases."
            },
            'n_restarts_optimizer': {
                'type': 'number_input',
                'label': 'Number of Optimizer Restarts',
                'min_value': 0,
                'max_value': 10,
                'value': 0,
                'step': 1,
                'help': "Number of times to restart the optimizer to escape local minima. 0 means use initial kernel parameters."
            },
            'normalize_y': {
                'type': 'checkbox',
                'label': 'Normalize Target Values',
                'value': False,
                'help': "Whether to normalize target values to zero mean and unit variance before fitting"
            }
        }
    
    def _get_kernel(self, kernel_type, length_scale):
        """Create the appropriate kernel based on the selected type"""
        if kernel_type == 'RBF':
            return RBF(length_scale=length_scale)
        elif kernel_type == 'Matern':
            return Matern(length_scale=length_scale)
        elif kernel_type == 'RationalQuadratic':
            return RationalQuadratic(length_scale=length_scale)
        elif kernel_type == 'ExpSineSquared':
            return ExpSineSquared(length_scale=length_scale)
        elif kernel_type == 'DotProduct':
            return DotProduct()
        elif kernel_type == 'RBF + White':
            return RBF(length_scale=length_scale) + WhiteKernel()
        elif kernel_type == 'Constant * RBF':
            return ConstantKernel() * RBF(length_scale=length_scale)
        else:
            return RBF(length_scale=length_scale)  # Default kernel
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Get parameters
        kernel_type = kwargs.get('kernel_type', 'RBF')
        length_scale = kwargs.get('length_scale', 1.0)
        alpha = kwargs.get('alpha', 1e-10)
        optimizer = kwargs.get('optimizer', 'fmin_l_bfgs_b')
        n_restarts_optimizer = kwargs.get('n_restarts_optimizer', 0)
        normalize_y = kwargs.get('normalize_y', False)
        
        # Create kernel
        kernel = self._get_kernel(kernel_type, length_scale)
        
        # Create and train model
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer if optimizer != 'None' else None,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            random_state=42
        )
        
        self.model.fit(X, y)
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 