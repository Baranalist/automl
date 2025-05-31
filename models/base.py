from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.target_name = None
        
    @abstractmethod
    def get_hyperparameters(self):
        """Return the model's hyperparameters for UI configuration"""
        pass
    
    @abstractmethod
    def train(self, X, y, **kwargs):
        """Train the model with given data and parameters"""
        pass
    
    def prepare_features(self, df, feature_cols, meta_info):
        """Prepare features for model training"""
        X = pd.DataFrame()
        
        for col in feature_cols:
            # Find metadata for this column
            col_meta = next((meta for meta in meta_info if meta["column"] == col), None)
            
            if col_meta["selected_type"] in ["int", "float"]:
                # Numeric columns are used as is
                X[col] = df[col]
            elif col_meta["selected_type"] == "str" and "categorical_type" in col_meta:
                if col_meta["categorical_type"] == "ordinal":
                    # Ordinal categorical columns are encoded using LabelEncoder
                    le = LabelEncoder()
                    X[col] = le.fit_transform(df[col])
                else:
                    # Nominal categorical columns are one-hot encoded
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    encoded = ohe.fit_transform(df[[col]])
                    # Create column names for the encoded features
                    feature_names = [f"{col}_{val}" for val in ohe.categories_[0]]
                    # Add encoded features to X
                    for i, name in enumerate(feature_names):
                        X[name] = encoded[:, i]
        
        return X
    
    def prepare_target(self, df, target_col, meta_info):
        """Prepare target variable for model training"""
        # Find metadata for target column
        target_meta = next((meta for meta in meta_info if meta["column"] == target_col), None)
        
        if target_meta["selected_type"] in ["int", "float"]:
            # Numeric target is used as is
            return df[target_col]
        elif target_meta["selected_type"] == "str" and "categorical_type" in target_meta:
            if target_meta["categorical_type"] == "ordinal":
                # Ordinal target is encoded using LabelEncoder
                le = LabelEncoder()
                return le.fit_transform(df[target_col])
            else:
                # Nominal target is encoded using LabelEncoder
                le = LabelEncoder()
                return le.fit_transform(df[target_col])
    
    def handle_missing_values(self, X, y):
        """Handle missing values in features and target"""
        # For features, fill missing values with mean for numeric columns
        # and mode for categorical columns
        for col in X.columns:
            if X[col].dtype in [np.int64, np.float64]:
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(X[col].mode()[0])
        
        # For target, drop rows with missing values
        mask = ~y.isna()
        return X[mask], y[mask]
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, random_state=random_state) 