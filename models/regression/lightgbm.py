from lightgbm import LGBMRegressor
import streamlit as st
from ..base import BaseModel

class LightGBMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LGBMRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'n_estimators': {
                'type': 'number_input',
                'label': 'Number of Trees',
                'min_value': 1,
                'max_value': 10000,
                'value': 100,
                'step': 10,
                'help': "Number of boosting iterations (trees) to perform. More trees generally give better performance but take longer to train."
            },
            'learning_rate': {
                'type': 'number_input',
                'label': 'Learning Rate',
                'min_value': 0.001,
                'max_value': 1.0,
                'value': 0.1,
                'step': 0.01,
                'help': "Controls shrinkage of each tree's contribution. Lower values require more trees but can lead to better generalization."
            },
            'boosting_type': {
                'type': 'selectbox',
                'label': 'Boosting Type',
                'options': ['gbdt', 'dart', 'goss', 'rf'],
                'help': "Type of boosting algorithm. 'gbdt' for traditional gradient boosting, 'dart' for dropout trees, 'goss' for gradient-based one-side sampling, 'rf' for random forest."
            },
            'objective': {
                'type': 'selectbox',
                'label': 'Objective Function',
                'options': ['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma', 'tweedie'],
                'help': "Loss function for regression. Choose based on your data distribution and requirements."
            },
            'max_depth': {
                'type': 'number_input',
                'label': 'Maximum Tree Depth',
                'min_value': -1,
                'max_value': 100,
                'value': -1,
                'step': 1,
                'help': "Maximum tree depth. -1 means no limit. Higher values make the model more complex and more likely to overfit."
            },
            'num_leaves': {
                'type': 'number_input',
                'label': 'Number of Leaves',
                'min_value': 2,
                'max_value': 256,
                'value': 31,
                'step': 1,
                'help': "Maximum number of leaves in one tree. Larger values make the model more complex."
            },
            'min_child_samples': {
                'type': 'number_input',
                'label': 'Minimum Child Samples',
                'min_value': 1,
                'max_value': 1000,
                'value': 20,
                'step': 1,
                'help': "Minimum number of samples required to form a leaf node."
            },
            'min_child_weight': {
                'type': 'number_input',
                'label': 'Minimum Child Weight',
                'min_value': 0.0,
                'max_value': 100.0,
                'value': 0.001,
                'step': 0.001,
                'help': "Minimum sum of instance weights needed in a child."
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
            'subsample_freq': {
                'type': 'number_input',
                'label': 'Subsample Frequency',
                'min_value': 0,
                'max_value': 100,
                'value': 0,
                'step': 1,
                'help': "Frequency of applying subsample. 0 means disable subsample."
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
                'min_value': 0.0,
                'max_value': 100.0,
                'value': 0.0,
                'step': 0.1,
                'help': "L1 regularization term on weights. Increasing this value makes the model more conservative."
            },
            'reg_lambda': {
                'type': 'number_input',
                'label': 'L2 Regularization',
                'min_value': 0.0,
                'max_value': 100.0,
                'value': 0.0,
                'step': 0.1,
                'help': "L2 regularization term on weights. Increasing this value makes the model more conservative."
            },
            'min_split_gain': {
                'type': 'number_input',
                'label': 'Minimum Split Gain',
                'min_value': 0.0,
                'max_value': 100.0,
                'value': 0.0,
                'step': 0.1,
                'help': "Minimum loss reduction required to make a split."
            },
            'max_bin': {
                'type': 'number_input',
                'label': 'Maximum Bins',
                'min_value': 1,
                'max_value': 512,
                'value': 255,
                'step': 1,
                'help': "Maximum number of bins for discretizing features."
            },
            'importance_type': {
                'type': 'selectbox',
                'label': 'Importance Type',
                'options': ['split', 'gain'],
                'help': "Method for feature importance calculation. 'split' for number of times a feature is used in a model, 'gain' for total gain of splits which use the feature."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'max_value': 1000,
                'value': 42,
                'step': 1,
                'help': "Random number seed for reproducibility."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Create a new dictionary for LightGBM parameters
        lgb_params = {}
        
        # Add all parameters that are valid for LGBMRegressor
        valid_params = [
            'n_estimators', 'learning_rate', 'boosting_type', 'objective',
            'max_depth', 'num_leaves', 'min_child_samples', 'min_child_weight',
            'subsample', 'subsample_freq', 'colsample_bytree', 'reg_alpha',
            'reg_lambda', 'min_split_gain', 'max_bin', 'importance_type',
            'random_state'
        ]
        
        for param in valid_params:
            if param in kwargs:
                lgb_params[param] = kwargs[param]
        
        self.model = LGBMRegressor(**lgb_params)
        self.model.fit(X, y)
        return self.model 