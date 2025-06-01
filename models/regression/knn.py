from sklearn.neighbors import KNeighborsRegressor
import streamlit as st
from ..base import BaseModel

class KNNRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = KNeighborsRegressor()
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        return {
            'n_neighbors': {
                'type': 'number_input',
                'label': 'Number of Neighbors',
                'min_value': 1,
                'max_value': 100,
                'value': 5,
                'step': 1,
                'help': "Number of neighbors to use for prediction. Larger values make the model more robust but less sensitive to local patterns."
            },
            'weights': {
                'type': 'selectbox',
                'label': 'Weight Function',
                'options': ['uniform', 'distance'],
                'help': "Weight function used in prediction. 'uniform' weights all neighbors equally, 'distance' weights points by inverse of their distance."
            },
            'algorithm': {
                'type': 'selectbox',
                'label': 'Algorithm',
                'options': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'help': "Algorithm used to compute nearest neighbors. 'auto' attempts to choose the best algorithm based on the data."
            },
            'leaf_size': {
                'type': 'number_input',
                'label': 'Leaf Size',
                'min_value': 1,
                'max_value': 100,
                'value': 30,
                'step': 1,
                'help': "Leaf size passed to BallTree or KDTree. Affects speed of construction and query. Smaller values mean deeper trees."
            },
            'p': {
                'type': 'number_input',
                'label': 'Minkowski Power',
                'min_value': 1,
                'max_value': 10,
                'value': 2,
                'step': 1,
                'help': "Power parameter for the Minkowski metric. p=1 for Manhattan distance, p=2 for Euclidean distance."
            },
            'metric': {
                'type': 'selectbox',
                'label': 'Distance Metric',
                'options': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
                'help': "Distance metric to use. 'minkowski' is the default, 'euclidean' is equivalent to minkowski with p=2, 'manhattan' with p=1."
            }
        }
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        self.model = KNeighborsRegressor(**kwargs)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 