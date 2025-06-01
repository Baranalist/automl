from catboost import CatBoostRegressor
import streamlit as st
from ..base import BaseModel

class CatBoostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = CatBoostRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'iterations': {
                'type': 'number_input',
                'label': 'Number of Trees',
                'min_value': 1,
                'max_value': 10000,
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
            'depth': {
                'type': 'number_input',
                'label': 'Tree Depth',
                'min_value': 1,
                'max_value': 16,
                'value': 6,
                'step': 1,
                'help': "Depth of each tree. Higher values make the model more complex and more likely to overfit."
            },
            'l2_leaf_reg': {
                'type': 'number_input',
                'label': 'L2 Regularization',
                'min_value': 0.0,
                'max_value': 100.0,
                'value': 3.0,
                'step': 0.1,
                'help': "L2 regularization term. Higher values make the model more conservative."
            },
            'loss_function': {
                'type': 'selectbox',
                'label': 'Loss Function',
                'options': ['RMSE', 'MAE', 'Quantile', 'Huber', 'LogLinQuantile', 'Poisson', 'MAPE', 'Tweedie'],
                'help': "Objective function to optimize. Choose based on your data distribution and requirements."
            },
            'bootstrap_type': {
                'type': 'selectbox',
                'label': 'Bootstrap Type',
                'options': ['Bayesian', 'Bernoulli', 'MVS', 'Poisson', 'No'],
                'help': "Bootstrapping method. 'Bayesian' for Bayesian bootstrap, 'Bernoulli' for Bernoulli bootstrap, 'MVS' for Minimal Variance Sampling, 'Poisson' for Poisson bootstrap, 'No' for no bootstrapping."
            },
            'subsample': {
                'type': 'number_input',
                'label': 'Subsample Ratio',
                'min_value': 0.1,
                'max_value': 1.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Sample ratio if bootstrapping is enabled. Not used with 'Bayesian' bootstrap type."
            },
            'colsample_bylevel': {
                'type': 'number_input',
                'label': 'Column Sample by Level',
                'min_value': 0.1,
                'max_value': 1.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Fraction of features to sample at each level. If smaller than 1.0, this results in random feature selection."
            },
            'random_strength': {
                'type': 'number_input',
                'label': 'Random Strength',
                'min_value': 0.0,
                'max_value': 100.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Used for score regularization of splits. Higher values make the model more random."
            },
            'bagging_temperature': {
                'type': 'number_input',
                'label': 'Bagging Temperature',
                'min_value': 0.0,
                'max_value': 10.0,
                'value': 1.0,
                'step': 0.1,
                'help': "For bootstrap_type='Bayesian', controls sampling intensity. Higher values make the sampling more random."
            },
            'border_count': {
                'type': 'number_input',
                'label': 'Border Count',
                'min_value': 1,
                'max_value': 255,
                'value': 32,
                'step': 1,
                'help': "Number of splits for numerical features. Higher values can lead to better precision but slower training."
            },
            'grow_policy': {
                'type': 'selectbox',
                'label': 'Grow Policy',
                'options': ['SymmetricTree', 'Depthwise', 'Lossguide'],
                'help': "Tree growth strategy. 'SymmetricTree' for symmetric trees, 'Depthwise' for depth-wise growth, 'Lossguide' for loss-guided growth."
            },
            'min_data_in_leaf': {
                'type': 'number_input',
                'label': 'Minimum Data in Leaf',
                'min_value': 1,
                'max_value': 1000,
                'value': 1,
                'step': 1,
                'help': "Minimum number of samples per leaf. Higher values make the model more conservative."
            },
            'max_leaves': {
                'type': 'number_input',
                'label': 'Maximum Leaves',
                'min_value': 1,
                'max_value': 64,
                'value': 31,
                'step': 1,
                'help': "Maximum number of leaves in a tree. Only used when grow_policy='Lossguide'."
            },
            'random_seed': {
                'type': 'number_input',
                'label': 'Random Seed',
                'min_value': 0,
                'max_value': 1000,
                'value': 42,
                'step': 1,
                'help': "Random number seed for reproducibility."
            },
            'od_type': {
                'type': 'selectbox',
                'label': 'Overfitting Detector Type',
                'options': ['IncToDec', 'Iter'],
                'help': "Type of overfitting detector. 'IncToDec' for increasing to decreasing, 'Iter' for iteration-based."
            },
            'od_wait': {
                'type': 'number_input',
                'label': 'Overfitting Detector Wait',
                'min_value': 1,
                'max_value': 100,
                'value': 10,
                'step': 1,
                'help': "Number of iterations to wait before early stopping if no improvement."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Create a new dictionary for CatBoost parameters
        cat_params = {}
        
        # Add all parameters that are valid for CatBoostRegressor
        valid_params = [
            'iterations', 'learning_rate', 'depth', 'l2_leaf_reg',
            'loss_function', 'bootstrap_type', 'colsample_bylevel',
            'random_strength', 'bagging_temperature', 'border_count',
            'grow_policy', 'min_data_in_leaf', 'max_leaves', 'random_seed',
            'od_type', 'od_wait'
        ]
        
        # Handle bootstrap type and subsample compatibility
        bootstrap_type = kwargs.get('bootstrap_type', 'Bayesian')
        if bootstrap_type != 'Bayesian' and 'subsample' in kwargs:
            cat_params['subsample'] = kwargs['subsample']
        
        # Add other parameters
        for param in valid_params:
            if param in kwargs:
                cat_params[param] = kwargs[param]
        
        self.model = CatBoostRegressor(**cat_params)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 