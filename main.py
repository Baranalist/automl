import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re
from functions import clean_and_convert_column
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from config.model_configs import REGRESSION_MODELS, CLASSIFICATION_MODELS, get_model_class

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
    meta_info = []
    
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
        
        # Add basic metadata for all columns
        meta_info.append({
            "column": col,
            "inferred_type": inferred_type,
            "selected_type": col_types[col],
            "unique_values": len(df[col].dropna().unique()),
            "missing_values": df[col].isna().sum()
        })

    # Ask about categorical string columns
    if str_columns:
        st.subheader("Categorical Column Details")
        for col in str_columns:
            cat_type = st.radio(f"Is '{col}' nominal or ordinal?", ["nominal", "ordinal"], horizontal=True, key=f"cat_{col}")

            if cat_type == "ordinal":
                unique_vals = list(df[col].dropna().unique())
                order = st.multiselect(f"Specify order for '{col}'", options=unique_vals, default=unique_vals, key=f"order_{col}")
                # Update metadata for ordinal columns
                for meta in meta_info:
                    if meta["column"] == col:
                        meta["categorical_type"] = "ordinal"
                        meta["ordinal_order"] = order
            else:
                # Update metadata for nominal columns
                for meta in meta_info:
                    if meta["column"] == col:
                        meta["categorical_type"] = "nominal"
                        meta["ordinal_order"] = None

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
                
                # Get model class and initialize model
                model_class = get_model_class(st.session_state.model_type, st.session_state.ml_task)
                model = model_class()
                model_params = {}
                
                # Advanced settings toggle
                if st.button("Advanced Settings"):
                    st.session_state.show_advanced = not st.session_state.show_advanced
                    st.rerun()
                
                if st.session_state.show_advanced:
                    st.write("Advanced Model Parameters")
                    hyperparameters = model.get_hyperparameters()

                    # Special handling for MLP Regression dynamic hidden layers
                    if st.session_state.model_type == "Multilayer Perceptron":
                        def update_num_layers():
                            st.session_state._rerun = True  # Dummy variable to force rerun

                        # Number of hidden layers input with callback
                        if "num_hidden_layers" not in st.session_state:
                            st.session_state["num_hidden_layers"] = 1

                        num_layers = st.number_input(
                            "Number of Hidden Layers",
                            min_value=1,
                            max_value=100,
                            value=st.session_state["num_hidden_layers"],
                            step=1,
                            key="num_hidden_layers",
                            on_change=update_num_layers
                        )

                        # Dynamically create keys and input fields for each layer
                        layer_sizes = []
                        for i in range(1, st.session_state["num_hidden_layers"] + 1):
                            key = f"layer_{i}_size"
                            if key not in st.session_state:
                                st.session_state[key] = 64 if i == 1 else 32  # default values
                            size = st.number_input(
                                f"Neurons in Layer {i}",
                                min_value=1,
                                max_value=1024,
                                value=st.session_state[key],
                                step=1,
                                key=key
                            )
                            layer_sizes.append(size)

                        # Remove extra keys if user reduces the number of layers
                        for i in range(st.session_state["num_hidden_layers"] + 1, 101):
                            key = f"layer_{i}_size"
                            if key in st.session_state:
                                del st.session_state[key]

                        # Save to model parameters
                        model_params["hidden_layer_sizes"] = tuple(layer_sizes)

                        # Add other hyperparameters as usual
                        for param_name, param_config in hyperparameters.items():
                            if param_name.startswith("layer_") or param_name == "num_hidden_layers":
                                continue  # Already handled
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
                    else:
                        # Display hyperparameters for all other models
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
                        y_pred = model.predict(X_test)
                        
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
        metrics_df = pd.DataFrame({
            'Metric': list(st.session_state.model_results['metrics'].keys()),
            'Value': list(st.session_state.model_results['metrics'].values())
        })
        st.dataframe(metrics_df)
        
        # Display coefficients/feature importance
        st.subheader("Feature Importance" if st.session_state.model_results['model_type'] == "Decision Tree" else "Model Coefficients")
        st.dataframe(st.session_state.model_results['feature_names'])
        
        # Plot semi-standardized residuals
        st.subheader("Semi-Standardized Residuals Plot")
        residuals = st.session_state.model_results['y_test'] - st.session_state.model_results['y_pred']
        std_residuals = residuals / np.std(residuals)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=st.session_state.model_results['y_pred'], y=std_residuals, ax=ax)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Semi-Standardized Residuals')
        ax.set_title('Semi-Standardized Residuals vs Predicted Values')
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

