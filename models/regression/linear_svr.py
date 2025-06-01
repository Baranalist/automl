from sklearn.svm import LinearSVR
import streamlit as st
from ..base import BaseModel

class LinearSVRModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LinearSVR()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'epsilon': {
                'type': 'number_input',
                'label': 'Epsilon',
                'min_value': 0.0,
                'max_value': 1.0,
                'value': 0.1,
                'step': 0.01,
                'help': "Width of the margin where no penalty is given during training. Specifies the margin of tolerance where no penalty is given to errors."
            },
            'loss': {
                'type': 'selectbox',
                'label': 'Loss Function',
                'options': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                'help': "Specifies the loss function. 'epsilon_insensitive' for standard epsilon-insensitive loss, 'squared_epsilon_insensitive' for squared epsilon-insensitive loss."
            },
            'dual': {
                'type': 'checkbox',
                'label': 'Use Dual Formulation',
                'value': True,
                'help': "Solve the dual problem. Set to False if n_samples > n_features for better performance."
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
            'C': {
                'type': 'number_input',
                'label': 'Regularization Parameter',
                'min_value': 0.01,
                'max_value': 100.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive."
            },
            'fit_intercept': {
                'type': 'checkbox',
                'label': 'Fit Intercept',
                'value': True,
                'help': "Whether to calculate the intercept for this model. If False, the data is assumed to be already centered."
            },
            'intercept_scaling': {
                'type': 'number_input',
                'label': 'Intercept Scaling',
                'min_value': 0.1,
                'max_value': 10.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Applies only when fit_intercept=True and dual=False. The constant added to the decision function."
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
        self.model = LinearSVR(**kwargs)
        self.model.fit(X, y)
        return self.model
        
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 