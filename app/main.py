import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_processing import clean_and_convert_column, get_column_metadata, update_categorical_metadata
from utils.visualization import plot_residuals, display_metrics, display_feature_importance
from config.model_configs import get_model_class, REGRESSION_MODELS, CLASSIFICATION_MODELS
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Initialize session state
if "converted_df" not in st.session_state:
    st.session_state.converted_df = None
if "meta_info" not in st.session_state:
    st.session_state.meta_info = None
if "conversion_done" not in st.session_state:
    st.session_state.conversion_done = False
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "ml_step" not in st.session_state:
    st.session_state.ml_step = 1
if "ml_task" not in st.session_state:
    st.session_state.ml_task = None
if "model_type" not in st.session_state:
    st.session_state.model_type = None
if "show_advanced" not in st.session_state:
    st.session_state.show_advanced = False

st.title("Data Type Validator & Converter")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)

    st.subheader("Initial Preview")
    st.dataframe(df.head())

    st.subheader("Set Column Types")
    col_types = {}
    str_columns = []
    
    for col in df.columns:
        inferred_type = pd.api.types.infer_dtype(df[col], skipna=True)
        options = ["int", "float", "date", "datetime", "str"]
        default = (
            "int" if "int" in inferred_type else
            "float" if "float" in inferred_type else
            "date" if "date" in inferred_type else
            "datetime" if "datetime" in inferred_type else
            "str"
        )
        col_types[col] = st.selectbox(f"{col} ({inferred_type})", options, index=options.index(default), key=f"type_{col}")
        if col_types[col] == "str":
            str_columns.append(col)
    
    # Get initial metadata
    meta_info = get_column_metadata(df, col_types)

    # Ask about categorical string columns
    st.subheader("Categorical Column Details")
    for col in str_columns:
        cat_type = st.radio(f"Is '{col}' nominal or ordinal?", ["nominal", "ordinal"], horizontal=True, key=f"cat_{col}")

        if cat_type == "ordinal":
            unique_vals = list(df[col].dropna().unique())
            order = st.multiselect(f"Specify order for '{col}'", options=unique_vals, default=unique_vals, key=f"order_{col}")
            meta_info = update_categorical_metadata(meta_info, col, cat_type, order)
        else:
            meta_info = update_categorical_metadata(meta_info, col, cat_type)

    # Show collected metadata
    st.subheader("Column Metadata Summary")
    meta_df = pd.DataFrame(meta_info)
    st.dataframe(meta_df)

    if st.button("Convert"):
        converted_df = df.copy()
        for col, target_type in col_types.items():
            try:
                converted_df[col] = clean_and_convert_column(converted_df[col], target_type)
            except Exception as e:
                st.warning(f"Could not convert column '{col}': {e}")

        st.subheader("Converted Data")
        st.dataframe(converted_df.head())
        
        # Store in session state
        st.session_state.converted_df = converted_df
        st.session_state.meta_info = meta_info
        st.session_state.conversion_done = True
        st.session_state.model_results = None
        st.session_state.ml_step = 1
        st.session_state.ml_task = None
        st.session_state.model_type = None
        st.session_state.show_advanced = False

# Machine Learning Section
if st.session_state.conversion_done:
    st.subheader("Machine Learning")
    
    # Step 1: Select ML Task
    if st.session_state.ml_step == 1:
        st.write("Step 1: Select Machine Learning Task")
        ml_task = st.radio("Select ML Task", ["Regression", "Classification"], horizontal=True)
        if st.button("Apply"):
            st.session_state.ml_task = ml_task
            st.session_state.ml_step = 2
            st.rerun()
    
    # Step 2: Select Model Type
    elif st.session_state.ml_step == 2:
        st.write("Step 2: Select Model Type")
        if st.session_state.ml_task == "Regression":
            model_type = st.selectbox(
                "Select Regression Model",
                list(REGRESSION_MODELS.keys())
            )
            st.write(REGRESSION_MODELS[model_type]["description"])
        
        if st.button("Apply"):
            st.session_state.model_type = model_type
            st.session_state.ml_step = 3
            st.rerun()
    
    # Step 3: Feature Selection and Model Training
    elif st.session_state.ml_step == 3:
        st.write("Step 3: Select Features and Train Model")
        
        # Get all columns that can be used for prediction
        available_columns = []
        for meta in st.session_state.meta_info:
            col = meta["column"]
            if meta["selected_type"] in ["int", "float"]:
                available_columns.append(col)
            elif meta["selected_type"] == "str" and "categorical_type" in meta:
                available_columns.append(col)
        
        if not available_columns:
            st.error("No suitable columns available. Please ensure you have numeric or categorical columns.")
        else:
            # Select target variable
            target_col = st.selectbox("Select Target Variable", available_columns)
            
            # Select predictor variables
            predictor_cols = st.multiselect(
                "Select Predictor Variables",
                [col for col in available_columns if col != target_col]
            )
            
            if predictor_cols:
                # Train-test split
                test_size = st.slider("Test Set Size (%)", 1, 99, 20) / 100
                
                # Get model class and hyperparameters
                model_class = get_model_class(st.session_state.model_type, st.session_state.ml_task)
                model = model_class()
                hyperparameters = model.get_hyperparameters()
                
                # Advanced settings toggle
                if st.button("Advanced Settings"):
                    st.session_state.show_advanced = not st.session_state.show_advanced
                    st.rerun()
                
                if st.session_state.show_advanced:
                    st.write("Advanced Model Parameters")
                    model_params = {}
                    
                    for param_name, param_config in hyperparameters.items():
                        if param_config['type'] == 'selectbox':
                            model_params[param_name] = st.selectbox(
                                param_config['label'],
                                param_config['options'],
                                help=param_config['help']
                            )
                        elif param_config['type'] == 'number_input':
                            model_params[param_name] = st.number_input(
                                param_config['label'],
                                min_value=param_config['min_value'],
                                value=param_config['value'],
                                help=param_config['help']
                            )
                        elif param_config['type'] == 'checkbox':
                            model_params[param_name] = st.checkbox(
                                param_config['label'],
                                value=param_config['value'],
                                help=param_config['help']
                            )
                        elif param_config['type'] == 'radio':
                            model_params[param_name] = st.radio(
                                param_config['label'],
                                param_config['options'],
                                horizontal=True,
                                help=param_config['help']
                            )
                        elif param_config['type'] == 'slider':
                            model_params[param_name] = st.slider(
                                param_config['label'],
                                min_value=param_config['min_value'],
                                max_value=param_config['max_value'],
                                value=param_config['value'],
                                step=param_config['step'],
                                help=param_config['help']
                            )
                
                if st.button("Apply"):
                    try:
                        # Prepare features and target
                        X = model.prepare_features(st.session_state.converted_df, predictor_cols, st.session_state.meta_info)
                        y = model.prepare_target(st.session_state.converted_df, target_col, st.session_state.meta_info)
                        
                        # Handle missing values
                        X, y = model.handle_missing_values(X, y)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=test_size)
                        
                        # Train model
                        model.train(X_train, y_train, **model_params if st.session_state.show_advanced else {})
                        
                        # Make predictions
                        y_pred = model.model.predict(X_test)
                        
                        # Calculate metrics
                        metrics = {
                            'MAE': mean_absolute_error(y_test, y_pred),
                            'MSE': mean_squared_error(y_test, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'MAPE (%)': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                        }
                        
                        # Store results in session state
                        st.session_state.model_results = {
                            'metrics': metrics,
                            'model': model.model,
                            'feature_names': X.columns,
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'model_type': st.session_state.model_type
                        }
                        
                        st.session_state.ml_step = 4
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during model training: {str(e)}")
    
    # Step 4: Display Results
    elif st.session_state.ml_step == 4 and st.session_state.model_results is not None:
        st.write("Step 4: Model Results")
        
        # Display metrics
        st.subheader("Model Performance Metrics")
        display_metrics(st.session_state.model_results['metrics'])
        
        # Display coefficients/feature importance
        st.subheader("Feature Importance" if st.session_state.model_results['model_type'] == "Decision Tree" else "Model Coefficients")
        display_feature_importance(
            st.session_state.model_results['model'],
            st.session_state.model_results['feature_names'],
            st.session_state.model_results['model_type']
        )
        
        # Plot residuals
        st.subheader("Semi-Standardized Residuals Plot")
        fig = plot_residuals(
            st.session_state.model_results['y_test'],
            st.session_state.model_results['y_pred']
        )
        st.pyplot(fig)
        
        if st.button("Start Over"):
            st.session_state.ml_step = 1
            st.session_state.ml_task = None
            st.session_state.model_type = None
            st.session_state.model_results = None
            st.session_state.show_advanced = False
            st.rerun()

    if st.session_state.converted_df is not None:
        st.download_button("Download Cleaned Data as CSV", st.session_state.converted_df.to_csv(index=False), "cleaned_data.csv", "text/csv") 