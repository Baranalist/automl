import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import streamlit as st
from ..base import BaseModel
import pandas as pd

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation='relu', output_activation='linear', dropout_rate=0.2, use_bn=True):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.use_bn = use_bn
        
        # Add input layer
        prev_size = input_size
        
        # Add hidden layers
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if use_bn:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        
        # Add output layer
        self.layers.append(nn.Linear(prev_size, 1))
        
        # Set activation functions
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation)
    
    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'elu': nn.ELU(),
            'linear': nn.Identity()
        }
        return activations.get(name, nn.ReLU())
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.use_bn:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        
        x = self.layers[-1](x)
        x = self.output_activation(x)
        return x
    
    def predict(self, x):
        """Make predictions using the model"""
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            return self(x)

class PyTorchRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        # Get the number of layers from session state if available
        n_layers = st.session_state.get('n_layers', 2) if 'st' in globals() else 2
        
        # Base hyperparameters that are always shown
        params = {
            'n_layers': {
                'type': 'number_input',
                'label': 'Number of Hidden Layers',
                'min_value': 1,
                'max_value': 5,
                'value': 2,
                'step': 1,
                'help': "Number of hidden layers in the neural network. More layers can learn more complex patterns but may overfit."
            }
        }
        
        # Add layer neuron inputs based on n_layers
        for i in range(1, n_layers + 1):
            params[f'layer_{i}_units'] = {
                'type': 'number_input',
                'label': f'Neurons in Layer {i}',
                'min_value': 8,
                'max_value': 1024,
                'value': 64 if i == 1 else 32,  # Default to 64 for first layer, 32 for others
                'step': 8,
                'help': f"Number of neurons in hidden layer {i}."
            }
        
        # Add remaining hyperparameters
        params.update({
            'activation': {
                'type': 'selectbox',
                'label': 'Hidden Layer Activation',
                'options': ['relu', 'tanh', 'sigmoid', 'elu'],
                'help': "Activation function for hidden layers. 'relu' is recommended for most cases."
            },
            'output_activation': {
                'type': 'selectbox',
                'label': 'Output Layer Activation',
                'options': ['linear', 'relu'],
                'help': "Activation function for output layer. 'linear' for general regression, 'relu' for non-negative outputs."
            },
            'dropout_rate': {
                'type': 'number_input',
                'label': 'Dropout Rate',
                'min_value': 0.0,
                'max_value': 0.5,
                'value': 0.2,
                'step': 0.1,
                'help': "Fraction of input units to drop. Helps prevent overfitting. 0.0 means no dropout."
            },
            'batch_normalization': {
                'type': 'checkbox',
                'label': 'Use Batch Normalization',
                'value': True,
                'help': "Whether to use batch normalization after each hidden layer. Helps with training stability."
            },
            'optimizer': {
                'type': 'selectbox',
                'label': 'Optimizer',
                'options': ['adam', 'rmsprop', 'sgd'],
                'help': "Optimization algorithm. 'adam' works well for most cases."
            },
            'learning_rate': {
                'type': 'number_input',
                'label': 'Learning Rate',
                'min_value': 0.00001,
                'max_value': 0.1,
                'value': 0.001,
                'step': 0.0001,
                'help': "Initial learning rate for the optimizer."
            },
            'batch_size': {
                'type': 'selectbox',
                'label': 'Batch Size',
                'options': ['32', '64', '128', '256'],
                'help': "Number of samples per gradient update. Larger values may speed up training but require more memory."
            },
            'epochs': {
                'type': 'number_input',
                'label': 'Maximum Epochs',
                'min_value': 10,
                'max_value': 1000,
                'value': 100,
                'step': 10,
                'help': "Maximum number of epochs to train. Training may stop earlier if validation loss stops improving."
            },
            'early_stopping_patience': {
                'type': 'number_input',
                'label': 'Early Stopping Patience',
                'min_value': 5,
                'max_value': 50,
                'value': 10,
                'step': 5,
                'help': "Number of epochs with no improvement after which training will be stopped."
            }
        })
        
        return params
    
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        # Convert DataFrame to numpy arrays and ensure float32 type
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.array(X, dtype=np.float32)
            
        if isinstance(y, pd.Series):
            y_np = y.values.astype(np.float32)
        else:
            y_np = np.array(y, dtype=np.float32)
        
        # Ensure y is 2D array
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        
        # Convert to PyTorch tensors
        X_tensor = torch.from_numpy(X_np)
        y_tensor = torch.from_numpy(y_np)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = int(kwargs.get('batch_size', '32'))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Get parameters
        n_layers = kwargs.get('n_layers', 2)
        hidden_sizes = []
        for i in range(1, n_layers + 1):
            units = kwargs.get(f'layer_{i}_units', 64 if i == 1 else 32)
            hidden_sizes.append(units)
        
        activation = kwargs.get('activation', 'relu')
        output_activation = kwargs.get('output_activation', 'linear')
        dropout_rate = kwargs.get('dropout_rate', 0.2)
        use_bn = kwargs.get('batch_normalization', True)
        
        # Create model
        self.model = NeuralNetwork(
            input_size=X_np.shape[1],
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=output_activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn
        ).to(self.device)
        
        # Configure optimizer
        optimizer_name = kwargs.get('optimizer', 'adam')
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:  # sgd
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        criterion = nn.MSELoss()
        epochs = kwargs.get('epochs', 100)
        early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        best_loss = float('inf')
        patience_counter = 0
        
        # Validation split
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_pred = self.model(X_batch)
                    val_loss += criterion(y_pred, y_batch).item()
            
            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.model.load_state_dict(best_model_state)
                    break
        
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Convert DataFrame to numpy array and ensure float32 type
        if isinstance(X, pd.DataFrame):
            X_np = X.values.astype(np.float32)
        else:
            X_np = np.array(X, dtype=np.float32)
        
        # Convert to PyTorch tensor and move to device
        X_tensor = torch.from_numpy(X_np).to(self.device)
        
        # Use the model's predict method
        predictions = self.model.predict(X_tensor)
        return predictions.cpu().numpy().flatten()  # Return 1D array 