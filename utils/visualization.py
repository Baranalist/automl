import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd

def plot_residuals(y_test, y_pred):
    """Plot semi-standardized residuals"""
    residuals = y_test - y_pred
    std_residuals = residuals / np.std(residuals)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=std_residuals, ax=ax)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Semi-Standardized Residuals')
    ax.set_title('Semi-Standardized Residuals vs Predicted Values')
    
    return fig

def display_metrics(metrics):
    """Display model performance metrics"""
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    st.dataframe(metrics_df)

def display_feature_importance(model, feature_names, model_type):
    """Display feature importance or coefficients"""
    if model_type == "Decision Tree":
        importance = model.feature_importances_
    else:
        importance = model.coef_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    st.dataframe(importance_df) 