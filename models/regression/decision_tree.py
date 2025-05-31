from sklearn.tree import DecisionTreeRegressor
import streamlit as st
from ..base import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'criterion': {
                'type': 'selectbox',
                'label': 'Criterion (Split Quality Measure)',
                'options': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                'help': "Function to measure split quality. 'squared_error' is most common for regression."
            },
            'splitter': {
                'type': 'selectbox',
                'label': 'Splitter Strategy',
                'options': ["best", "random"],
                'help': "Strategy to choose the split. 'best' chooses the best split, 'random' chooses the best random split."
            },
            'max_depth': {
                'type': 'number_input',
                'label': 'Maximum Tree Depth',
                'min_value': 1,
                'value': 5,
                'help': "Maximum depth of the tree. Higher values may lead to overfitting."
            },
            'min_samples_split': {
                'type': 'number_input',
                'label': 'Minimum Samples Split',
                'min_value': 2,
                'value': 2,
                'help': "Minimum number of samples required to split an internal node."
            },
            'min_samples_leaf': {
                'type': 'number_input',
                'label': 'Minimum Samples Leaf',
                'min_value': 1,
                'value': 1,
                'help': "Minimum number of samples required to be at a leaf node."
            },
            'min_weight_fraction_leaf': {
                'type': 'slider',
                'label': 'Minimum Weight Fraction Leaf',
                'min_value': 0.0,
                'max_value': 0.5,
                'value': 0.0,
                'step': 0.01,
                'help': "Minimum weighted fraction of samples required at a leaf node."
            },
            'max_features': {
                'type': 'selectbox',
                'label': 'Maximum Features',
                'options': ["None", "sqrt", "log2", "auto"],
                'help': "Strategy for considering features when splitting."
            },
            'max_leaf_nodes': {
                'type': 'number_input',
                'label': 'Maximum Leaf Nodes',
                'min_value': 2,
                'value': None,
                'help': "Maximum number of leaf nodes in the tree. None for unlimited."
            },
            'min_impurity_decrease': {
                'type': 'number_input',
                'label': 'Minimum Impurity Decrease',
                'min_value': 0.0,
                'value': 0.0,
                'step': 0.001,
                'help': "Minimum impurity decrease required for a split."
            },
            'ccp_alpha': {
                'type': 'number_input',
                'label': 'CCP Alpha (Pruning Parameter)',
                'min_value': 0.0,
                'value': 0.0,
                'step': 0.001,
                'help': "Complexity parameter for Minimal Cost-Complexity Pruning."
            },
            'random_state': {
                'type': 'number_input',
                'label': 'Random State',
                'min_value': 0,
                'value': 42,
                'help': "Seed for reproducibility. Set to None for random behavior."
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