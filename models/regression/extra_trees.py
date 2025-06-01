from sklearn.ensemble import ExtraTreesRegressor
import streamlit as st
from ..base import BaseModel

class ExtraTreesModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = ExtraTreesRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'n_estimators': {
                'type': 'number_input',
                'label': 'Number of Trees',
                'min_value': 1,
                'value': 100,
                'step': 10,
                'help': "Number of trees in the forest. More trees generally give better performance but take longer to train."
            },
            'criterion': {
                'type': 'selectbox',
                'label': 'Split Criterion',
                'options': ['squared_error', 'absolute_error', 'poisson', 'friedman_mse'],
                'help': "Function to measure the quality of a split. 'squared_error' is for mean squared error, 'absolute_error' for mean absolute error, 'poisson' for Poisson deviance, and 'friedman_mse' for Friedman's mean squared error."
            },
            'max_depth': {
                'type': 'number_input',
                'label': 'Maximum Tree Depth',
                'min_value': 1,
                'value': None,
                'help': "Maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples."
            },
            'min_samples_split': {
                'type': 'number_input',
                'label': 'Minimum Samples for Split',
                'min_value': 2,
                'value': 2,
                'help': "Minimum number of samples required to split an internal node."
            },
            'min_samples_leaf': {
                'type': 'number_input',
                'label': 'Minimum Samples per Leaf',
                'min_value': 1,
                'value': 1,
                'help': "Minimum number of samples required to be at a leaf node."
            },
            'min_weight_fraction_leaf': {
                'type': 'number_input',
                'label': 'Minimum Weight Fraction per Leaf',
                'min_value': 0.0,
                'max_value': 0.5,
                'value': 0.0,
                'step': 0.1,
                'help': "Minimum weighted fraction of the sum total of weights required to be at a leaf node."
            },
            'max_features': {
                'type': 'selectbox',
                'label': 'Maximum Features',
                'options': ['sqrt', 'log2', None],
                'help': "Number of features to consider when looking for the best split. 'sqrt' uses sqrt(n_features), 'log2' uses log2(n_features), and None uses all features."
            },
            'max_leaf_nodes': {
                'type': 'number_input',
                'label': 'Maximum Leaf Nodes',
                'min_value': 2,
                'value': None,
                'help': "Grow trees with max_leaf_nodes in best-first fashion. If None, then unlimited number of leaf nodes."
            },
            'min_impurity_decrease': {
                'type': 'number_input',
                'label': 'Minimum Impurity Decrease',
                'min_value': 0.0,
                'value': 0.0,
                'step': 0.1,
                'help': "A node will be split if this split induces a decrease of the impurity greater than or equal to this value."
            },
            'bootstrap': {
                'type': 'checkbox',
                'label': 'Bootstrap Samples',
                'value': False,
                'help': "Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree."
            },
            'oob_score': {
                'type': 'checkbox',
                'label': 'Out-of-Bag Score',
                'value': False,
                'help': "Whether to use out-of-bag samples to estimate the generalization accuracy. Only available if bootstrap=True."
            },
            'n_jobs': {
                'type': 'number_input',
                'label': 'Number of Jobs',
                'min_value': -1,
                'value': None,
                'help': "Number of jobs to run in parallel. -1 means using all processors."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': 42,
                'help': "Controls the randomness of the bootstrapping of the samples used when building trees."
            },
            'warm_start': {
                'type': 'checkbox',
                'label': 'Warm Start',
                'value': False,
                'help': "When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble."
            },
            'ccp_alpha': {
                'type': 'number_input',
                'label': 'CCP Alpha',
                'min_value': 0.0,
                'value': 0.0,
                'step': 0.1,
                'help': "Complexity parameter used for Minimal Cost-Complexity Pruning."
            },
            'max_samples': {
                'type': 'number_input',
                'label': 'Maximum Samples',
                'min_value': 0.0,
                'max_value': 1.0,
                'value': None,
                'step': 0.1,
                'help': "If bootstrap is True, the number of samples to draw from X to train each base estimator."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Create a new dictionary for ExtraTrees parameters
        et_params = {}
        
        # Add all parameters that are valid for ExtraTreesRegressor
        valid_params = [
            'n_estimators', 'criterion', 'max_depth', 'min_samples_split',
            'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
            'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score',
            'n_jobs', 'random_state', 'warm_start', 'ccp_alpha', 'max_samples'
        ]
        
        for param in valid_params:
            if param in kwargs:
                et_params[param] = kwargs[param]
        
        self.model = ExtraTreesRegressor(**et_params)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 