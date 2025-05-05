from learnware.learnware import Learnware
from .hetero_market import HeteroLearnwareMarket
from ..benchmarks import split_feature

import torch
import numpy as np
from typing import List

def learnware_reuse(
    learnware: Learnware,
    learnware_block_assignment: List,
    learnware_market: HeteroLearnwareMarket,
    x_user: np.ndarray,
    user_feature_block_assignment: List,
    dim_list: List,
    task_type: str,
    device: torch.device,
    y_user: np.ndarray,
):
    # transform x_user to torch
    x_user = torch.tensor(x_user, dtype=torch.float32).to(device)

    # split the x_user
    x_user_list = split_feature(
        X=x_user,
        dim_list=dim_list,
        feature_blocks_assignment=user_feature_block_assignment,
    )

    # get embedding
    x_user_emb = learnware_market._get_embedding(x_user, user_feature_block_assignment)

    fill_block_ids = []
    for idx in learnware_block_assignment:
        if idx not in user_feature_block_assignment:
            x_user_list[idx] = learnware_market._reconstruct_embedding(x_user_emb, idx)
            fill_block_ids.append(idx)

    # combine the x_user_list on the learnware feature space
    x_user_rec_list = [x_user_list[i] for i in learnware_block_assignment]
    x_user_rec = torch.cat(x_user_rec_list, dim=1)

    # move x_user_rec to cpu and numpy
    x_user_rec = x_user_rec.cpu().detach().numpy()

    # make prediction
    y_learnware = learnware.predict(x_user_rec)

    # make prediction on the system engine
    y_engine = learnware_market.system_engine(x_user_emb).cpu().detach().numpy()

    # ensemble the results
    if task_type == "regression":
        # Ensemble the outputs by averaging for regression
        y_engine = y_engine.squeeze()
        predictions = [y_engine, y_learnware]
        y_ensemble = np.mean(predictions, axis=0)
        predictions_with_user = [y_user, y_learnware]
        y_ensemble_with_user = np.mean(predictions_with_user, axis=0)
    elif task_type == "classification":
        # Average the probabilities and then take argmax
        classes = learnware.model.model.classes_
        classes = np.array([int(i) for i in classes])
        y_engine = y_engine[:, classes]
        predictions = [y_engine, y_learnware]
        y_ensemble = np.mean(predictions, axis=0)
        y_ensemble = np.argmax(y_ensemble, axis=-1)
        predictions_with_user = [y_user, y_learnware]
        y_ensemble_with_user = np.mean(predictions_with_user, axis=0)
        y_ensemble_with_user = np.argmax(y_ensemble_with_user, axis=-1)
    else:
        raise ValueError(
            "Unsupported task type. Choose 'classification' or 'regression'."
        )

    if task_type == "classification":
        results = {
            "y_engine": np.argmax(y_engine, axis=-1),
            "y_learnware": np.argmax(y_learnware, axis=-1),
            "y_ensemble": y_ensemble,
            "y_ensemble_with_user": y_ensemble_with_user,
        }
    else:
        results = {
            "y_engine": y_engine,
            "y_learnware": y_learnware,
            "y_ensemble": y_ensemble,
            "y_ensemble_with_user": y_ensemble_with_user,
        }

    return results
