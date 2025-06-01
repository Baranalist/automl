from sklearn.neural_network import MLPRegressor
import streamlit as st
from ..base import BaseModel

class MLPRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = MLPRegressor()
        self.num_hidden_layers = 1  # Default number of hidden layers
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        # First, get the number of hidden layers from the user
        num_layers = st.number_input(
            "Number of Hidden Layers",
            min_value=1,
            max_value=5,
            value=self.num_hidden_layers,
            step=1,
            help="Number of hidden layers in the neural network"
        )
        
        # Update the number of hidden layers
        self.num_hidden_layers = num_layers
        
        # Create base hyperparameters
        hyperparameters = {
            'num_hidden_layers': {
                'type': 'number_input',
                'label': 'Number of Hidden Layers',
                'min_value': 1,
                'max_value': 5,
                'value': self.num_hidden_layers,
                'step': 1,
                'help': "Number of hidden layers in the neural network"
            }
        }
        
        # Add dynamic layer size inputs
        for i in range(self.num_hidden_layers):
            layer_key = f'layer_{i+1}_size'
            hyperparameters[layer_key] = {
                'type': 'number_input',
                'label': f'Neurons in Layer {i+1}',
                'min_value': 1,
                'max_value': 1000,
                'value': 100 if i == 0 else 50,  # Default values
                'step': 1,
                'help': f"Number of neurons in hidden layer {i+1}"
            }
        
        # Add other hyperparameters
        hyperparameters.update({
            'activation': {
                'type': 'selectbox',
                'label': 'Activation Function',
                'options': ['relu', 'tanh', 'logistic'],
                'help': "Activation function for the hidden layers. 'relu' is recommended for most cases."
            },
            'solver': {
                'type': 'selectbox',
                'label': 'Optimizer',
                'options': ['adam', 'lbfgs', 'sgd'],
                'help': "Algorithm for weight optimization. 'adam' works well for most cases, 'lbfgs' for smaller datasets."
            },
            'alpha': {
                'type': 'number_input',
                'label': 'L2 Regularization',
                'min_value': 0.0001,
                'max_value': 1.0,
                'value': 0.0001,
                'step': 0.0001,
                'help': "L2 regularization term. Larger values mean stronger regularization."
            },
            'batch_size': {
                'type': 'selectbox',
                'label': 'Batch Size',
                'options': ['auto', '32', '64', '128', '256'],
                'help': "Size of minibatches for stochastic optimizers. 'auto' uses min(200, n_samples)."
            },
            'learning_rate': {
                'type': 'selectbox',
                'label': 'Learning Rate Schedule',
                'options': ['constant', 'adaptive', 'invscaling'],
                'help': "Learning rate schedule for weight updates. Only used with 'sgd' solver."
            },
            'learning_rate_init': {
                'type': 'number_input',
                'label': 'Initial Learning Rate',
                'min_value': 0.0001,
                'max_value': 1.0,
                'value': 0.001,
                'step': 0.0001,
                'help': "Initial learning rate. Only used with 'sgd' or 'adam' solver."
            },
            'max_iter': {
                'type': 'number_input',
                'label': 'Maximum Iterations',
                'min_value': 100,
                'max_value': 10000,
                'value': 1000,
                'step': 100,
                'help': "Maximum number of iterations. The solver iterates until convergence or this number of iterations."
            },
            'early_stopping': {
                'type': 'checkbox',
                'label': 'Early Stopping',
                'value': False,
                'help': "Whether to use early stopping to terminate training when validation score is not improving."
            },
            'validation_fraction': {
                'type': 'number_input',
                'label': 'Validation Fraction',
                'min_value': 0.1,
                'max_value': 0.3,
                'value': 0.1,
                'step': 0.05,
                'help': "Proportion of training data to set aside as validation set for early stopping."
            },
            'n_iter_no_change': {
                'type': 'number_input',
                'label': 'Patience',
                'min_value': 5,
                'max_value': 50,
                'value': 10,
                'step': 5,
                'help': "Maximum number of epochs with no improvement to wait before stopping when early_stopping=True."
            }
        })
        
        return hyperparameters
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Create a new dictionary for MLP parameters
        mlp_params = {}
        
        # Construct hidden_layer_sizes from individual layer sizes
        hidden_layer_sizes = []
        for i in range(self.num_hidden_layers):
            layer_key = f'layer_{i+1}_size'
            if layer_key in kwargs and kwargs[layer_key] is not None:
                hidden_layer_sizes.append(int(kwargs[layer_key]))
        
        if hidden_layer_sizes:
            mlp_params['hidden_layer_sizes'] = tuple(hidden_layer_sizes)
        
        # Convert batch_size from string to int or 'auto'
        if 'batch_size' in kwargs:
            batch_size = kwargs['batch_size']
            if batch_size and batch_size != 'auto':
                try:
                    batch_size = int(batch_size)
                    mlp_params['batch_size'] = batch_size
                except (ValueError, TypeError):
                    mlp_params['batch_size'] = 'auto'
            else:
                mlp_params['batch_size'] = 'auto'
        
        # Add all other parameters that are valid for MLPRegressor
        valid_params = [
            'activation', 'solver', 'alpha', 'learning_rate',
            'learning_rate_init', 'max_iter', 'early_stopping',
            'validation_fraction', 'n_iter_no_change'
        ]
        
        # Add parameters
        for param in valid_params:
            if param in kwargs and kwargs[param] is not None:
                mlp_params[param] = kwargs[param]
        
        self.model = MLPRegressor(**mlp_params)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        return self.model.predict(X) 