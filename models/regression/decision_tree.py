from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from ..base import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'criterion': {
                'type': 'selectbox',
                'label': 'Criterion',
                'options': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                'help': "The function to measure the quality of a split."
            },
            'splitter': {
                'type': 'selectbox',
                'label': 'Splitter',
                'options': ['best', 'random'],
                'help': "The strategy used to choose the split at each node."
            },
            'max_depth': {
                'type': 'number_input',
                'label': 'Max Depth',
                'min_value': 1,
                'max_value': 100,
                'value': None,
                'help': "The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure."
            },
            'min_samples_split': {
                'type': 'number_input',
                'label': 'Min Samples Split',
                'min_value': 2,
                'value': 2,
                'help': "The minimum number of samples required to split an internal node."
            },
            'min_samples_leaf': {
                'type': 'number_input',
                'label': 'Min Samples Leaf',
                'min_value': 1,
                'value': 1,
                'help': "The minimum number of samples required to be at a leaf node."
            },
            'min_weight_fraction_leaf': {
                'type': 'number_input',
                'label': 'Min Weight Fraction Leaf',
                'min_value': 0.0,
                'max_value': 0.5,
                'value': 0.0,
                'help': "The minimum weighted fraction of the sum total of weights required to be at a leaf node."
            },
            'max_features': {
                'type': 'selectbox',
                'label': 'Max Features',
                'options': ['None', 'sqrt', 'log2'],
                'help': "The number of features to consider when looking for the best split."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': 42,
                'help': "Controls the randomness of the estimator."
            },
            'max_leaf_nodes': {
                'type': 'number_input',
                'label': 'Max Leaf Nodes',
                'min_value': 2,
                'value': None,
                'help': "Grow a tree with max_leaf_nodes in best-first fashion."
            },
            'min_impurity_decrease': {
                'type': 'number_input',
                'label': 'Min Impurity Decrease',
                'min_value': 0.0,
                'value': 0.0,
                'help': "A node will be split if this split induces a decrease of the impurity greater than or equal to this value."
            },
            'ccp_alpha': {
                'type': 'number_input',
                'label': 'CCP Alpha',
                'min_value': 0.0,
                'value': 0.0,
                'help': "Complexity parameter used for Minimal Cost-Complexity Pruning."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Convert string "None" to Python None for max_features
        if 'max_features' in kwargs and kwargs['max_features'] == "None":
            kwargs['max_features'] = None
        
        self.model = DecisionTreeRegressor(**kwargs)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 