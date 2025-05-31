from sklearn.ensemble import GradientBoostingRegressor
import streamlit as st
from ..base import BaseModel

class GradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'loss': {
                'type': 'selectbox',
                'label': 'Loss Function',
                'options': ['squared_error', 'absolute_error', 'huber', 'quantile'],
                'help': "Loss function to be optimized. 'squared_error' for mean squared error, 'absolute_error' for mean absolute error, 'huber' for Huber loss, and 'quantile' for quantile regression."
            },
            'learning_rate': {
                'type': 'number_input',
                'label': 'Learning Rate',
                'min_value': 0.001,
                'value': 0.1,
                'step': 0.01,
                'help': "Learning rate shrinks the contribution of each tree. Lower values require more trees but can lead to better generalization."
            },
            'n_estimators': {
                'type': 'number_input',
                'label': 'Number of Trees',
                'min_value': 1,
                'value': 100,
                'step': 10,
                'help': "Number of boosting stages (trees) to perform. More trees generally give better performance but take longer to train."
            },
            'subsample': {
                'type': 'number_input',
                'label': 'Subsample Ratio',
                'min_value': 0.1,
                'max_value': 1.0,
                'value': 1.0,
                'step': 0.1,
                'help': "Fraction of samples used for fitting the individual base learners. If smaller than 1.0, this results in stochastic gradient boosting."
            },
            'criterion': {
                'type': 'selectbox',
                'label': 'Split Criterion',
                'options': ['friedman_mse', 'squared_error'],
                'help': "Function to measure the quality of a split. 'friedman_mse' uses Friedman's mean squared error, 'squared_error' uses mean squared error."
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
            'max_depth': {
                'type': 'number_input',
                'label': 'Maximum Tree Depth',
                'min_value': 1,
                'value': 3,
                'help': "Maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree."
            },
            'min_impurity_decrease': {
                'type': 'number_input',
                'label': 'Minimum Impurity Decrease',
                'min_value': 0.0,
                'value': 0.0,
                'step': 0.1,
                'help': "A node will be split if this split induces a decrease of the impurity greater than or equal to this value."
            },
            'max_features': {
                'type': 'selectbox',
                'label': 'Maximum Features',
                'options': ['sqrt', 'log2', None],
                'help': "Number of features to consider when looking for the best split. 'sqrt' uses sqrt(n_features), 'log2' uses log2(n_features), and None uses all features."
            },
            'alpha': {
                'type': 'number_input',
                'label': 'Alpha',
                'min_value': 0.0,
                'max_value': 1.0,
                'value': 0.9,
                'step': 0.1,
                'help': "The alpha-quantile of the Huber loss function and the quantile loss function. Only used if loss='huber' or loss='quantile'."
            },
            'max_leaf_nodes': {
                'type': 'number_input',
                'label': 'Maximum Leaf Nodes',
                'min_value': 2,
                'value': None,
                'help': "Grow trees with max_leaf_nodes in best-first fashion. If None, then unlimited number of leaf nodes."
            },
            'warm_start': {
                'type': 'checkbox',
                'label': 'Warm Start',
                'value': False,
                'help': "When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble."
            },
            'validation_fraction': {
                'type': 'number_input',
                'label': 'Validation Fraction',
                'min_value': 0.0,
                'max_value': 1.0,
                'value': 0.1,
                'step': 0.1,
                'help': "Proportion of training data to set aside as validation set for early stopping."
            },
            'n_iter_no_change': {
                'type': 'number_input',
                'label': 'Iterations Without Change',
                'min_value': 1,
                'value': None,
                'help': "Used to decide if early stopping will be used to terminate training when validation score is not improving."
            },
            'tol': {
                'type': 'number_input',
                'label': 'Tolerance',
                'min_value': 0.0,
                'value': 1e-4,
                'step': 1e-4,
                'help': "Tolerance for the early stopping. When the loss is not improving by at least tol for n_iter_no_change consecutive iterations, training stops."
            },
            'ccp_alpha': {
                'type': 'number_input',
                'label': 'CCP Alpha',
                'min_value': 0.0,
                'value': 0.0,
                'step': 0.1,
                'help': "Complexity parameter used for Minimal Cost-Complexity Pruning."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': 42,
                'help': "Controls the random seed given to the base estimator at each boosting iteration."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Create a new dictionary for GradientBoosting parameters
        gb_params = {}
        
        # Add all parameters that are valid for GradientBoostingRegressor
        valid_params = [
            'loss', 'learning_rate', 'n_estimators', 'subsample', 'criterion',
            'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf',
            'max_depth', 'min_impurity_decrease', 'max_features', 'alpha',
            'max_leaf_nodes', 'warm_start', 'validation_fraction',
            'n_iter_no_change', 'tol', 'ccp_alpha', 'random_state'
        ]
        
        for param in valid_params:
            if param in kwargs:
                # Handle special cases for parameters
                if param == 'max_features' and kwargs[param] == 'auto':
                    gb_params[param] = 'sqrt'  # 'auto' is deprecated, use 'sqrt' instead
                elif param == 'max_depth' and kwargs[param] is None:
                    gb_params[param] = 3  # Default value for max_depth
                elif param == 'n_iter_no_change' and kwargs[param] is None:
                    gb_params[param] = 10  # Default value for n_iter_no_change
                else:
                    gb_params[param] = kwargs[param]
        
        self.model = GradientBoostingRegressor(**gb_params)
        self.model.fit(X, y)
        return self.model 