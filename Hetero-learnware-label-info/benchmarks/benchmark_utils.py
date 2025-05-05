import os

from learnware.learnware import Learnware, get_stat_spec_from_config
from learnware.specification import Specification
from learnware.config import C
from learnware.utils import read_yaml_to_dict
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
from typing import Union
import torch
import math

def get_learnware_from_dirpath(learnware_dirpath: str):
    from ..core.specification import (
        LabeledSpecification,
        LabeledSpecificationCLS,
        LabeledRKMESpecification,
    )
    learnware_yaml_path = os.path.join(
        learnware_dirpath, C.learnware_folder_config["yaml_file"]
    )
    learnware_config = read_yaml_to_dict(learnware_yaml_path)

    if "module_path" not in learnware_config["model"]:
        learnware_config["model"]["module_path"] = C.learnware_folder_config[
            "module_file"
        ]

    learnware_spec = Specification()
    for _stat_spec in learnware_config["stat_specifications"]:
        stat_spec = _stat_spec.copy()
        stat_spec_path = os.path.join(learnware_dirpath, stat_spec["file_name"])

        stat_spec["file_name"] = stat_spec_path
        if stat_spec["class_name"] == "LabeledSpecificationCLS":
            stat_spec_inst = LabeledSpecificationCLS()
            stat_spec_inst.load(stat_spec_path)
        elif stat_spec["class_name"] == "LabeledSpecification":
            stat_spec_inst = LabeledSpecification()
            stat_spec_inst.load(stat_spec_path)
        elif stat_spec["class_name"] == "LabeledRKMESpecification":
            stat_spec_inst = LabeledRKMESpecification()
            stat_spec_inst.load(stat_spec_path)
        else:
            stat_spec_inst = get_stat_spec_from_config(stat_spec)
        learnware_spec.update_stat_spec(**{stat_spec_inst.type: stat_spec_inst})

    return Learnware(
        id=os.path.basename(learnware_dirpath),
        model=learnware_config["model"],
        specification=learnware_spec,
        learnware_dirpath=learnware_dirpath,
    )


def convert_to_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Converts input array to a numpy array.

    Parameters:
    array (Union[np.ndarray, torch.Tensor]): Input array which can be either a numpy array or a PyTorch tensor.

    Returns:
    np.ndarray: Converted numpy array.

    Raises:
    TypeError: If the input array type is not supported.
    """
    if isinstance(array, np.ndarray):
        return array
    try:
        return array.detach().cpu().numpy()  # Assuming array is a PyTorch tensor.
    except AttributeError:
        print(f"Current array: {array}")
        raise TypeError(
            f"Unsupported type for array. Expected numpy.ndarray or torch.Tensor, currently got {type(array)}."
        )


def evaluate_loss(
    task_type: str,
    y_true: Union[np.ndarray, torch.Tensor],
    y_predict: Union[np.ndarray, torch.Tensor],
    model_classes: list = None,
) -> None:
    """
    Evaluates and prints the loss based on the task type, and handles predictions that are labels or probabilities.
    Supports numpy arrays and tensors as input formats.

    Parameters:
    task_type (str): Type of task, "classification" or "regression".
    y_true (Union[np.ndarray, torch.Tensor]): True labels or values.
    y_predict (Union[np.ndarray, torch.Tensor]): Predicted probabilities (for classification) or predicted values (for regression).

    Returns:
    None: This function only prints the loss metrics (accuracy for classification or RMSE for regression).

    Raises:
    ValueError: If the task type is unsupported or the prediction shape is unexpected in classification.
    """
    y_true = convert_to_numpy(y_true)
    y_predict = convert_to_numpy(y_predict)

    if task_type == "classification":
        if y_predict.ndim == 2 and y_predict.shape[1] > 1:
            # y_predict is a probability matrix
            y_pred_labels = np.argmax(y_predict, axis=1)
            y_pred_labels = [int(i) for i in y_pred_labels]
            if model_classes is not None:
                y_pred_labels = [model_classes[i] for i in y_pred_labels]
            y_pred_labels = np.array(y_pred_labels)
        elif y_predict.ndim == 1 or (y_predict.ndim == 2 and y_predict.shape[1] == 1):
            # y_predict is class labels
            y_pred_labels = np.squeeze(y_predict)
        else:
            raise ValueError("Unexpected shape for y_predict in classification.")

        accuracy = accuracy_score(y_true, y_pred_labels)

        return accuracy
    elif task_type == "regression":
        rmse = math.sqrt(mean_squared_error(y_true, y_predict))
        return rmse
    else:
        raise ValueError(
            "Unsupported task type. `task_type` must be 'classification' or 'regression'."
        )
