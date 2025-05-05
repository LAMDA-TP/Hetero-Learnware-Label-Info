""" modified from https://github.com/clabrugere/pytorch-scarf/blob/master/example/utils.py
"""

import random

import numpy as np
import torch
from tqdm.auto import tqdm
from torch import optim

from .loss import (
    NTXent,
    reconstruction_loss,
    supervised_loss,
    self_supervised_contrastive_loss,
    supervised_contrastive_loss,
    similarity_loss,
)
from torch.utils.data import DataLoader
from torch import nn
from typing import List
from ...benchmarks import split_feature


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.has_mps:  # support for apple M1 chips or more advanced
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def train_epoch(
    model: nn.Module,
    system_engine: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    task_type: str,
    loss_weights: dict = None,
) -> float:
    model.train()
    system_engine.train()
    epoch_loss = 0.0
    ntxent = NTXent()

    if loss_weights is None:
        loss_weights = {"weight_cont": 1.0, "weight_rec": 1.0, "weight_supervised": 1.0}

    for x, y, weights in train_loader:
        x, y_true, weights = x.to(device), y.to(device), weights.to(device)

        emb_anchor, emb_positive, x_rec = model(x)
        y_pred = system_engine(emb_anchor)

        loss_cont = ntxent(emb_anchor, emb_positive, weights) * loss_weights.get(
            "weight_cont", 1.0
        )
        loss_rec = reconstruction_loss(x, x_rec, weights) * loss_weights.get(
            "weight_rec", 1.0
        )
        loss_supervised = supervised_loss(
            y_pred, y_true, task_type, weights
        ) * loss_weights.get("weight_supervised", 1.0)

        total_loss = loss_cont + loss_rec + loss_supervised

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    return epoch_loss / len(train_loader.dataset)


def train_epoch_multi(
    models: List[nn.Module],
    system_engine: nn.Module,
    train_loaders: List[DataLoader],
    optimizer: optim.Optimizer,
    device: torch.device,
    task_type: str,
    learnware_feature_split_info: dict,
    loss_weights: dict = None,
    mmd_calculator_paras: dict = None,
) -> float:
    # Ensure each model and its corresponding DataLoader are in training mode
    for model in models:
        model.train()
    system_engine.train()

    # Set default weights if not provided
    if loss_weights is None:
        loss_weights = {
            "weight_cont": 1.0,
            "weight_rec": 1.0,
            "weight_supervised": 1.0,
            "weight_global": 0.1,
            "use_weight": True,
        }

    if mmd_calculator_paras is None:
        mmd_calculator_paras = {
            "task_type": task_type,
            "gamma": 0.1,
            "cuda_idx": 0,
            "cond_mmd_coef": 1,
            "num_bins": 10,
        }

    total_epoch_loss = 0.0
    total_samples = 0

    for i, train_loader in enumerate(train_loaders):
        for x, y, weights in train_loader:
            bs_loss = 0
            x, y_true, weights = x.to(device), y.to(device), weights.to(device)
            feature_blocks_assignment = learnware_feature_split_info[
                "feature_block_assignments"
            ][i]

            x_list = split_feature(
                X=x,
                dim_list=learnware_feature_split_info["dim_list"],
                feature_blocks_assignment=feature_blocks_assignment,
            )
            temp_x_emb_list = []
            for j in feature_blocks_assignment:
                temp_x = x_list[j]
                # Obtain embeddings and reconstructed outputs
                temp_x_emb, temp_x_rec = models[j](temp_x)
                # calculate the reconstruction loss
                if loss_weights.get("weight_rec", 0) > 0:
                    loss_rec = reconstruction_loss(
                        temp_x, temp_x_rec, weights
                    ) * loss_weights.get("weight_rec")
                else:
                    loss_rec = 0
                bs_loss += loss_rec

                temp_x_emb_list.append(temp_x_emb)
            # calculate the contrastive loss
            temp_x_emb_stack = torch.stack(temp_x_emb_list, dim=1)
            if loss_weights.get("weight_cont", 0) > 0:
                loss_cont = self_supervised_contrastive_loss(
                    features=temp_x_emb_stack,
                    weights=weights,
                    device=device,
                    use_weight=loss_weights.get("use_weight"),
                ) * loss_weights.get("weight_cont")
            else:
                loss_cont = 0
            bs_loss += loss_cont

            if loss_weights.get("weight_cont_supervised", 0) > 0:
                loss_cont_sup = supervised_contrastive_loss(
                    features=temp_x_emb_stack,
                    weights=weights,
                    labels=y_true,
                    device=device,
                    use_weight=loss_weights.get("use_weight"),
                ) * loss_weights.get("weight_cont_supervised")
            else:
                loss_cont_sup = 0
            bs_loss += loss_cont_sup

            # calculate the supervised loss
            temp_x_emb_mean = torch.mean(temp_x_emb_stack, dim=1)
            y_pred = system_engine(temp_x_emb_mean)
            if loss_weights.get("weight_supervised", 0) > 0:
                loss_supervised = supervised_loss(
                    y_pred, y_true, task_type, weights
                ) * loss_weights.get("weight_supervised")
            else:
                loss_supervised = 0
            bs_loss += loss_supervised

            # calculate the difference loss
            if loss_weights.get("weight_global", 0) > 0:
                loss_diff = similarity_loss(
                    temp_x_emb_list, weights
                ) * loss_weights.get("weight_global")
            else:
                loss_diff = 0
            bs_loss += loss_diff

            # Loss backpropagation and optimizer step
            if bs_loss != 0:
                optimizer.zero_grad()
                bs_loss.backward()
                optimizer.step()

                # Accumulate loss for averaging later
                total_epoch_loss += bs_loss.item()
                total_samples += len(weights)

    return total_epoch_loss / total_samples


def dataset_embeddings(model, loader, device):
    embeddings = []

    for x, y, weights in tqdm(loader):
        x = x.to(device)
        embeddings.append(model.get_embeddings(x))

    embeddings = torch.cat(embeddings).cpu().numpy()

    return embeddings


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
