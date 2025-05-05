import joblib
import random
import torch
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import os

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train_lgb_model(X, y, task_type, save_path=None, regenerate=False):
    """
    If a model exists at save_path, it is loaded and returned. Otherwise, trains a new LightGBM model.
    """
    # Load model if exists
    if save_path is not None and os.path.exists(save_path) and not regenerate:
        return joblib.load(save_path)

    # Define parameter grid
    lgb_params_list = [
        [0.015, 224, 66],
        [0.005, 300, 50],
        [0.01, 128, 80],
        [0.15, 224, 80],
        [0.01, 300, 66],
    ]

    # Base LightGBM parameters
    lgb_params = {
        "boosting_type": "gbdt",
        "learning_rate": None,
        "num_leaves": None,
        "max_depth": None,
        "n_estimators": 1000,
        "boost_from_average": False,
        "verbose": -1,
    }

    if task_type == "classification":
        classes = np.unique(y)
        if len(classes) == 2:
            lgb_params["objective"] = "binary"
            lgb_params["metric"] = "binary_logloss"
        else:
            lgb_params["objective"] = "multiclass"
            lgb_params["metric"] = "multi_logloss"
        model_class = lgb.LGBMClassifier
        evaluate = accuracy_score
    elif task_type == "regression":
        lgb_params["objective"] = "regression"
        lgb_params["metric"] = "rmse"
        model_class = lgb.LGBMRegressor
        evaluate = lambda true, pred: np.sqrt(mean_squared_error(true, pred))  # RMSE
    else:
        raise ValueError("task_type must be 'classification' or 'regression'.")

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.33, random_state=0, shuffle=True
    )

    # Training and validation
    best_param, best_score = None, None
    for param in lgb_params_list:
        lgb_params["learning_rate"] = param[0]
        lgb_params["num_leaves"] = param[1]
        lgb_params["max_depth"] = param[2]

        model = model_class(**lgb_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        score = evaluate(y_val, y_pred)
        print(
            f"learning_rate={param[0]}, num_leaves={param[1]}, max_depth={param[2]}, score={score}"
        )

        if (
            best_score is None
            or (task_type == "classification" and score > best_score)
            or (task_type == "regression" and score < best_score)
        ):
            best_param, best_score = param, score

    # Train the final model with the best parameters found
    lgb_params["learning_rate"] = best_param[0]
    lgb_params["num_leaves"] = best_param[1]
    lgb_params["max_depth"] = best_param[2]
    model = model_class(**lgb_params)
    model.fit(X, y)

    # Save model if a path is provided
    if save_path is not None:
        joblib.dump(model, save_path)

    return model