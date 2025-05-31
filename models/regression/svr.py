from sklearn.svm import SVR
import streamlit as st
from ..base import BaseModel

class SVRModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SVR()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'kernel': {
                'type': 'selectbox',
                'label': 'Kernel Type',
                'options': ['linear', 'poly', 'rbf', 'sigmoid'],
                'help': "Specifies the kernel type. 'linear' for linear kernel, 'poly' for polynomial kernel, 'rbf' for radial basis function kernel, 'sigmoid' for sigmoid kernel."
            },
            'C': {
                'type': 'number_input',
                'label': 'Regularization Parameter',
                'min_value': 0.01,
                'max_value': 100.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive."
            },
            'epsilon': {
                'type': 'number_input',
                'label': 'Epsilon',
                'min_value': 0.0,
                'max_value': 1.0,
                'value': 0.1,
                'step': 0.01,
                'help': "Epsilon-tube within which no penalty is given in training loss. Specifies the margin of tolerance where no penalty is given to errors."
            },
            'degree': {
                'type': 'number_input',
                'label': 'Polynomial Degree',
                'min_value': 1,
                'max_value': 10,
                'value': 3,
                'step': 1,
                'help': "Degree of the polynomial kernel function. Only used when kernel='poly'."
            },
            'gamma': {
                'type': 'selectbox',
                'label': 'Kernel Coefficient',
                'options': ['scale', 'auto'],
                'help': "Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. 'scale' uses 1 / (n_features * X.var()) as value, 'auto' uses 1 / n_features."
            },
            'coef0': {
                'type': 'number_input',
                'label': 'Kernel Coefficient 0',
                'min_value': 0.0,
                'max_value': 10.0,
                'value': 0.0,
                'step': 0.1,
                'help': "Independent term in kernel function. Only used when kernel='poly' or kernel='sigmoid'."
            },
            'shrinking': {
                'type': 'checkbox',
                'label': 'Use Shrinking Heuristic',
                'value': True,
                'help': "Whether to use the shrinking heuristic. This can speed up training but may not always be optimal."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Tolerance',
                'min_value': 0.00001,
                'max_value': 0.1,
                'value': 0.001,
                'step': 0.0001,
                'help': "Tolerance for stopping criterion. The optimization will stop when the change in the objective function is less than this value."
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 100,
                'max_value': 10000,
                'value': 1000,
                'step': 100,
                'help': "Hard limit on iterations. -1 means no limit, but we restrict it to a reasonable range for better control."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Create a new dictionary for SVR parameters
        svr_params = {}
        
        # Add all parameters that are valid for SVR
        valid_params = [
            'kernel', 'C', 'epsilon', 'degree', 'gamma',
            'coef0', 'shrinking', 'tol', 'max_iter'
        ]
        
        # Add parameters
        for param in valid_params:
            if param in kwargs:
                svr_params[param] = kwargs[param]
        
        self.model = SVR(**svr_params)
        self.model.fit(X, y)
        return self.model 