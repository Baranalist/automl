"""Model configurations for the application."""

REGRESSION_MODELS = {
    "Linear Regression": {
        "module_path": "models.regression.linear",
        "class_name": "LinearRegressionModel",
        "description": "Simple linear regression model for predicting continuous values."
    },
    "Decision Tree": {
        "module_path": "models.regression.decision_tree",
        "class_name": "DecisionTreeModel",
        "description": "Decision tree model for regression tasks with feature importance visualization."
    },
    "Ridge Regression": {
        "module_path": "models.regression.ridge",
        "class_name": "RidgeRegressionModel",
        "description": "Linear regression with L2 regularization to prevent overfitting."
    },
    "Lasso Regression": {
        "module_path": "models.regression.lasso",
        "class_name": "LassoRegressionModel",
        "description": "Linear regression with L1 regularization for feature selection."
    },
    "ElasticNet": {
        "module_path": "models.regression.elastic_net",
        "class_name": "ElasticNetModel",
        "description": "Linear regression with both L1 and L2 regularization, combining Ridge and Lasso."
    },
    "Bayesian Ridge": {
        "module_path": "models.regression.bayesian_ridge",
        "class_name": "BayesianRidgeModel",
        "description": "Bayesian ridge regression that estimates regularization parameters automatically."
    },
    "Huber Regression": {
        "module_path": "models.regression.huber",
        "class_name": "HuberModel",
        "description": "Robust regression that is less sensitive to outliers by using a combination of squared and absolute errors."
    },
    "Theil-Sen Regression": {
        "module_path": "models.regression.theil_sen",
        "class_name": "TheilSenModel",
        "description": "Robust regression that uses the median of slopes between all pairs of points, making it highly resistant to outliers."
    },
    "RANSAC Regression": {
        "module_path": "models.regression.ransac",
        "class_name": "RANSACModel",
        "description": "Robust regression that fits a model to random subsets of the data and identifies inliers, making it highly resistant to outliers."
    },
    "Poisson Regression": {
        "module_path": "models.regression.poisson",
        "class_name": "PoissonRegressionModel",
        "description": "Generalized linear model for count data, using a Poisson distribution. Suitable for modeling count or frequency data."
    },
    "Tweedie Regression": {
        "module_path": "models.regression.tweedie",
        "class_name": "TweedieRegressionModel",
        "description": "Generalized linear model with Tweedie distribution, supporting various distributions including Normal, Poisson, Gamma, and Inverse Gaussian through the power parameter."
    },
    "Quantile Regression": {
        "module_path": "models.regression.quantile",
        "class_name": "QuantileRegressionModel",
        "description": "Linear regression that predicts conditional quantiles of the target variable, useful for understanding the full distribution of the response variable."
    },
    "Random Forest": {
        "module_path": "models.regression.random_forest",
        "class_name": "RandomForestModel",
        "description": "An ensemble learning method that operates by constructing multiple decision trees and outputting the mean prediction of the individual trees. Good for handling non-linear relationships and feature interactions."
    },
    "Extra Trees": {
        "module_path": "models.regression.extra_trees",
        "class_name": "ExtraTreesModel",
        "description": "An ensemble learning method similar to Random Forest but with more randomization in the tree building process. It uses random splits for all features at each node, which can lead to more diverse trees and potentially better generalization."
    },
    "Gradient Boosting": {
        "module_path": "models.regression.gradient_boosting",
        "class_name": "GradientBoostingModel",
        "description": "A powerful ensemble learning method that builds trees sequentially, where each new tree helps to correct errors made by previously trained trees. Good for handling complex non-linear relationships and often provides high accuracy."
    },
    "XGBoost": {
        "module_path": "models.regression.xgboost",
        "class_name": "XGBoostModel",
        "description": "An optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. It implements machine learning algorithms under the Gradient Boosting framework, providing parallel tree boosting and is widely used for structured/tabular data."
    },
    "LightGBM": {
        "module_path": "models.regression.lightgbm",
        "class_name": "LightGBMModel",
        "description": "A fast, distributed, high-performance gradient boosting framework based on decision tree algorithms. It's designed to be efficient and scalable, with support for categorical features and optimized for memory usage."
    },
    "CatBoost": {
        "module_path": "models.regression.catboost",
        "class_name": "CatBoostModel",
        "description": "A high-performance gradient boosting library that handles categorical features automatically. It's known for its excellent performance on tabular data and its ability to handle categorical features without preprocessing."
    },
    "Histogram Gradient Boosting": {
        "module_path": "models.regression.hist_gradient_boosting",
        "class_name": "HistGradientBoostingModel",
        "description": "A fast, memory-efficient gradient boosting implementation that uses histogram-based algorithms. It's particularly efficient for large datasets and can handle categorical features automatically."
    },
    "Support Vector Regression": {
        "module_path": "models.regression.svr",
        "class_name": "SVRModel",
        "description": "A support vector machine for regression tasks. It finds a function that deviates from the observed values by a value no greater than a specified margin while being as flat as possible. Good for non-linear regression problems."
    },
    "Linear Support Vector Regression": {
        "module_path": "models.regression.linear_svr",
        "class_name": "LinearSVRModel",
        "description": "A linear support vector machine for regression tasks. It's faster than the standard SVR for large datasets and is particularly effective when the number of features is greater than the number of samples."
    },
    "K-Nearest Neighbors Regression": {
        "module_path": "models.regression.knn",
        "class_name": "KNNRegressionModel",
        "description": "A non-parametric regression method that predicts values based on the average of the k nearest neighbors. It's simple, intuitive, and works well for small to medium-sized datasets with clear local patterns."
    },
    "Multilayer Perceptron": {
        "module_path": "models.regression.mlp",
        "class_name": "MLPRegressionModel",
        "description": "A neural network model that can learn complex non-linear relationships. It consists of multiple layers of neurons and can approximate any continuous function. Good for complex regression problems with sufficient training data."
    },
    "Neural Network (PyTorch)": {
        "module_path": "models.regression.pytorch",
        "class_name": "PyTorchRegressionModel",
        "description": "A deep neural network implemented using PyTorch. It can learn complex non-linear relationships and includes features like dropout and batch normalization for better generalization. Good for complex regression problems with sufficient training data."
    },
    "Gaussian Process": {
        "module_path": "models.regression.gaussian_process",
        "class_name": "GaussianProcessRegressionModel",
        "description": "A Gaussian Process Regression model that uses kernel functions to model the covariance between data points. This model is particularly good at capturing uncertainty in predictions and can model complex non-linear relationships. It works well with small to medium-sized datasets and provides both predictions and uncertainty estimates."
    },
    "ARD Regression": {
        "module_path": "models.regression.ard",
        "class_name": "ARDRegressionModel",
        "description": "Automatic Relevance Determination (ARD) Regression is a Bayesian linear regression model that automatically determines the relevance of features. It uses a hierarchical prior over the weights and can effectively perform feature selection by setting irrelevant features' coefficients to zero. Good for datasets with many features where you want to identify the most important ones."
    },
    "Isotonic Regression": {
        "module_path": "models.regression.isotonic",
        "class_name": "IsotonicRegressionModel",
        "description": "A non-parametric regression model that fits a non-decreasing (or non-increasing) function to the data. It's particularly useful when you know the relationship between features and target should be monotonic. The model can handle both increasing and decreasing relationships and provides options for handling out-of-bounds predictions."
    },
    "PLS Regression": {
        "module_path": "models.regression.pls",
        "class_name": "PLSRegressionModel",
        "description": "Partial Least Squares (PLS) Regression is a dimensionality reduction technique that finds a linear subspace that captures maximum variance in both features and target variables. It's particularly useful for datasets with many features and potential multicollinearity. The model can handle both single and multiple target variables."
    },
    "LARS": {
        "module_path": "models.regression.lars",
        "class_name": "LARSRegressionModel",
        "description": "Least Angle Regression (LARS) is an efficient algorithm for computing the entire path of coefficients for linear regression. It's particularly useful for feature selection and can handle both L1 and L2 regularization. The model provides a way to select the optimal number of features and can be used for both regression and classification tasks."
    },
    "LassoLARS": {
        "module_path": "models.regression.lasso_lars",
        "class_name": "LassoLARSModel",
        "description": "LassoLARS is a linear model that combines the L1 regularization of Lasso with the Least Angle Regression (LARS) algorithm. It's particularly useful for feature selection and handling multicollinearity. The model can automatically select the most relevant features while maintaining good predictive performance."
    },
    "Generalized Additive Model": {
        "module_path": "models.regression.gam",
        "class_name": "GAMModel",
        "description": "Generalized Additive Models (GAMs) are a flexible class of models that combine the properties of generalized linear models with additive models. They can capture non-linear relationships between features and the target variable using smooth functions (splines). GAMs are particularly useful when you suspect non-linear relationships in your data but want to maintain interpretability."
    }
}

CLASSIFICATION_MODELS = {
    # Add classification models here
}

def get_model_class(model_name, task_type):
    """Get the model class based on the model name and task type."""
    if task_type == "Regression":
        if model_name not in REGRESSION_MODELS:
            raise ValueError(f"Unknown regression model: {model_name}")
        model_config = REGRESSION_MODELS[model_name]
    elif task_type == "Classification":
        if model_name not in CLASSIFICATION_MODELS:
            raise ValueError(f"Unknown classification model: {model_name}")
        model_config = CLASSIFICATION_MODELS[model_name]
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    module = __import__(model_config["module_path"], fromlist=[model_config["class_name"]])
    return getattr(module, model_config["class_name"]) 