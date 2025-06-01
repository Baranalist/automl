from xgboost import XGBRegressor
import streamlit as st
from ..base import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = XGBRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'n_estimators': {
                'type': 'number_input',
                'label': 'Number of Trees',
                'min_value': 1,
                'value': 100,
                'step': 10,
                'help': "Number of boosting rounds (trees) to perform. More trees generally give better performance but take longer to train."
            },
            'learning_rate': {
                'type': 'number_input',
                'label': 'Learning Rate',
                'min_value': 0.001,
                'max_value': 1.0,
                'value': 0.1,
                'step': 0.01,
                'help': "Step size shrinkage to prevent overfitting. Lower values require more trees but can lead to better generalization."
            },
            'max_depth': {
                'type': 'number_input',
                'label': 'Maximum Tree Depth',
                'min_value': 1,
                'value': 3,
                'help': "Maximum depth of a tree. Increasing this value makes the model more complex and more likely to overfit."
            },
            'min_child_weight': {
                'type': 'number_input',
                'label': 'Minimum Child Weight',
                'min_value': 0,
                'value': 1,
                'help': "Minimum sum of instance weight (hessian) needed in a child. Higher values make the algorithm more conservative."
            },
            'gamma': {
                'type': 'number_input',
                'label': 'Gamma',
                'min_value': 0,
                'value': 0,
                'step': 0.1,
                'help': "Minimum loss reduction required to make a split. Higher values make the algorithm more conservative."
            },
            'subsample': {
                'type': 'number_input',
                'label': 'Subsample Ratio',
                'min_value': 0.1,
                'max_value': 1.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Fraction of samples used per tree. If smaller than 1.0, this results in stochastic gradient boosting."
            },
            'colsample_bytree': {
                'type': 'number_input',
                'label': 'Column Sample by Tree',
                'min_value': 0.1,
                'max_value': 1.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Fraction of features used per tree. If smaller than 1.0, this results in random feature selection."
            },
            'reg_alpha': {
                'type': 'number_input',
                'label': 'L1 Regularization',
                'min_value': 0,
                'value': 0,
                'step': 0.1,
                'help': "L1 regularization term on weights. Increasing this value makes the model more conservative."
            },
            'reg_lambda': {
                'type': 'number_input',
                'label': 'L2 Regularization',
                'min_value': 0,
                'value': 1,
                'step': 0.1,
                'help': "L2 regularization term on weights. Increasing this value makes the model more conservative."
            },
            'booster': {
                'type': 'selectbox',
                'label': 'Booster Type',
                'options': ['gbtree', 'gblinear', 'dart'],
                'help': "Type of booster to use. 'gbtree' for tree-based models, 'gblinear' for linear models, 'dart' for dropout trees."
            },
            'tree_method': {
                'type': 'selectbox',
                'label': 'Tree Method',
                'options': ['auto', 'exact', 'approx', 'hist'],
                'help': "Tree construction algorithm. 'auto' uses the best method based on the data size."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': 42,
                'help': "Random number seed for reproducibility."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Create a new dictionary for XGBoost parameters
        xgb_params = {}
        
        # Add all parameters that are valid for XGBRegressor
        valid_params = [
            'n_estimators', 'learning_rate', 'max_depth', 'min_child_weight',
            'gamma', 'subsample', 'colsample_bytree', 'reg_alpha', 'reg_lambda',
            'booster', 'tree_method', 'random_state'
        ]
        
        for param in valid_params:
            if param in kwargs:
                xgb_params[param] = kwargs[param]
        
        self.model = XGBRegressor(**xgb_params)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 