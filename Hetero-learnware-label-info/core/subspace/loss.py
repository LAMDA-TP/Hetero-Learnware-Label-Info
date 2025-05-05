import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class NTXent(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z_i: torch.Tensor, z_j: torch.Tensor, weights: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute weighted NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.Tensor): Anchor batch of samples.
            z_j (torch.Tensor): Positive batch of samples.
            weights (torch.Tensor, optional): Weights for each pair of samples, defaults to None.

        Returns:
            torch.Tensor: The computed loss.
        """
        batch_size = z_i.size(0)

        # Compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(2 * batch_size, device=z_i.device, dtype=torch.bool)).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))

        # Check if weights are provided
        if weights is not None:
            # Ensure the weights are of the correct shape
            weights = weights.repeat(2)  # since each sample and its pair are considered
            weighted_losses = all_losses * weights
            loss = torch.sum(weighted_losses) / torch.sum(weights)
        else:
            loss = torch.mean(all_losses)

        return loss


def reconstruction_loss(
    x: torch.Tensor, x_rec: torch.Tensor, weights: torch.Tensor = None
) -> torch.Tensor:
    """Compute the weighted reconstruction loss between the original and the reconstructed samples.
       If weights are provided, the loss is weighted by the sample weights.

    Args:
        x (torch.Tensor): Original samples.
        x_rec (torch.Tensor): Reconstructed samples.
        weights (torch.Tensor, optional): Weights for each sample, defaults to None.

    Returns:
        torch.Tensor: The computed loss.
    """
    # Calculate the mean squared error for each sample, summing over features
    mse = F.mse_loss(x_rec, x, reduction="none").mean(dim=1)

    # Check if weights are provided
    if weights is not None:
        # Ensure weights are reshaped to match the dimension of mse
        weights = weights.view(-1)
        # Apply weights: multiply the mse by the weights
        weighted_mse = mse * weights
        # Calculate the mean of the weighted mse losses
        loss = torch.sum(weighted_mse) / torch.sum(weights)
    else:
        # Calculate the mean of the mse losses when no weights are provided
        loss = torch.mean(mse)

    return loss


def supervised_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    task_type: str,
    weights: torch.Tensor = None,
) -> torch.Tensor:
    """Compute the supervised loss using either MSE or cross-entropy, potentially weighted by sample weights.

    Args:
        y_pred (torch.Tensor): Predicted labels or values.
        y_true (torch.Tensor): True labels or values.
        task_type (str): Type of task ('classification' or 'regression').
        weights (torch.Tensor, optional): Weights for each sample, defaults to None.

    Returns:
        torch.Tensor: Computed loss.
    """
    if task_type == "regression":
        loss = F.mse_loss(y_pred, y_true, reduction="none")
        if weights is not None:
            # Ensure weights are reshaped to match the dimension of loss
            weights = weights.view(-1)
            loss = loss * weights
            return torch.sum(loss) / torch.sum(weights)
        else:
            return torch.mean(loss)
    elif task_type == "classification":
        loss = F.cross_entropy(y_pred, y_true, reduction="none")
        if weights is not None:
            # Ensure weights are reshaped to match the dimension of loss
            weights = weights.view(-1)
            loss = loss * weights
            return torch.sum(loss) / torch.sum(weights)
        else:
            return torch.mean(loss)
    else:
        raise ValueError(
            f"Task type {task_type} not supported. Use 'classification' or 'regression'"
        )


class MMDLoss(nn.Module):
    def __init__(self, gamma: float = 0.1) -> None:
        """Initialize the MMD loss module with a specified gamma for the RBF kernel.

        Args:
            gamma (float, optional): Bandwidth for the Gaussian kernel. Defaults to 0.1.
        """
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
        beta_1: torch.Tensor,
        beta_2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Maximum-Mean-Discrepancy (MMD) distance between two sets of embeddings.

        Args:
            z_1 (torch.Tensor): Embeddings from the first distribution.
            z_2 (torch.Tensor): Embeddings from the second distribution.
            beta_1 (torch.Tensor): Weights for embeddings in the first distribution.
            beta_2 (torch.Tensor): Weights for embeddings in the second distribution.

        Returns:
            torch.Tensor: The computed MMD distance.
        """
        return self._mmd_distance(z_1, z_2, beta_1, beta_2)

    def _torch_rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel using torch operations."""
        x1 = x1.double()
        x2 = x2.double()
        X12norm = (
            torch.sum(x1**2, 1, keepdim=True)
            - 2 * x1 @ x2.T
            + torch.sum(x2**2, 1, keepdim=True).T
        )
        return torch.exp(-X12norm * self.gamma)

    def _inner_product(
        self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
        beta_1: torch.Tensor,
        beta_2: torch.Tensor,
    ) -> float:
        """Calculate the inner product using the RBF kernel."""
        beta_1 = beta_1.reshape(1, -1).double()
        beta_2 = beta_2.reshape(1, -1).double()
        Z1 = z_1.double().reshape(z_1.shape[0], -1)
        Z2 = z_2.double().reshape(z_2.shape[0], -1)
        v = torch.sum(self._torch_rbf_kernel(Z1, Z2) * (beta_1.T @ beta_2))
        return float(v)

    def _mmd_distance(
        self,
        z_1: torch.Tensor,
        z_2: torch.Tensor,
        beta_1: torch.Tensor,
        beta_2: torch.Tensor,
    ) -> float:
        """Compute MMD using the computed inner products."""
        term1 = self._inner_product(z_1, z_1, beta_1, beta_1)
        term2 = self._inner_product(z_1, z_2, beta_1, beta_2)
        term3 = self._inner_product(z_2, z_2, beta_2, beta_2)
        return term1 - 2 * term2 + term3


def self_supervised_contrastive_loss(
    features: torch.Tensor,
    weights: torch.Tensor,
    device,
    temperature=10,
    base_temperature=10,
    use_weight=False,
):
    """
    Compute the self-supervised VPCL loss.

    Parameters
    ----------
    features : torch.Tensor
        The encoded features of multiple partitions of input tables, with shape (bs, n_partition, proj_dim).

    Returns
    -------
    torch.Tensor
        The computed self-supervised VPCL loss.
    """
    batch_size = features.shape[0]
    labels = torch.arange(batch_size, dtype=torch.long, device=device).view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    contrast_count = features.shape[1]
    # [[0,1],[2,3]] -> [0,2,1,3]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature
    )
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    mask = mask.repeat(anchor_count, contrast_count)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0,
    )
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute weighted mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # Scaling loss according to temperature and weights
    loss = -(temperature / base_temperature) * mean_log_prob_pos

    if use_weight:
        loss = loss.view(anchor_count, batch_size)
        loss = loss * weights
        loss = loss.mean()
        loss = loss / (anchor_count * (anchor_count - 1) * batch_size)
    else:
        loss = loss.view(anchor_count, batch_size).mean()
        loss = loss / (anchor_count * batch_size * (anchor_count - 1))

    return loss


def supervised_contrastive_loss(
    features: torch.Tensor,
    weights: torch.Tensor,
    labels: torch.Tensor,
    device,
    temperature=10,
    base_temperature=10,
    use_weight=False,
):
    """Compute the supervised VPCL loss.

    Parameters
    ----------
    features: torch.Tensor
        the encoded features of multiple partitions of input tables, with shape ``(bs, n_partition, proj_dim)``.

    labels: torch.Tensor
        the class labels to be used for building positive/negative pairs in VPCL.

    Returns
    -------
    loss: torch.Tensor
        the computed VPCL loss.

    """
    labels = labels.contiguous().view(-1, 1)
    batch_size = features.shape[0]
    mask = torch.eq(labels, labels.T).float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

    # contrast_mode == 'all'
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature
    )
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
        0,
    )
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # compute mean of log-likelihood over positive
    loss = -(temperature / base_temperature) * mean_log_prob_pos

    if use_weight:
        loss = loss.view(anchor_count, batch_size)
        loss = loss * weights
        loss = loss.mean()
        loss = loss / (anchor_count * (anchor_count - 1) * batch_size)
    else:
        loss = loss.view(anchor_count, batch_size).mean()
        loss = loss / (
            anchor_count * batch_size * (anchor_count - 1)
        )  # keep it a constant,  n_eq = torch.sum(mask) is bad because of the inconsistant value for different batch

    return loss


def similarity_loss(
    x_list: List[torch.Tensor], weights: torch.Tensor = None
) -> torch.Tensor:
    # Stack the list of tensors to form a batch along the task dimension
    x_stack = torch.stack(x_list, dim=1)
    # Calculate the mean of the stacked tensors along the task dimension
    x_mean = torch.mean(x_stack, dim=1)
    # Calculate the mean squared error between each tensor in the stack and the mean
    mse = F.mse_loss(x_stack, x_mean.unsqueeze(1).expand_as(x_stack), reduction="none")
    # Sum the MSE across the dimension feature
    mse = mse.sum(dim=2)

    # Check if weights are provided
    if weights is not None:
        # Ensure weights are reshaped to match the dimension of mse
        weights = weights.view(-1)
        # Apply weights: multiply the mse by the weights and then sum
        weighted_mse = (mse * weights.unsqueeze(1)).sum(dim=0)
        # Calculate the mean of the weighted mse losses
        loss = torch.sum(weighted_mse) / torch.sum(weights)
    else:
        # Calculate the mean of the mse losses when no weights are provided
        loss = torch.mean(mse)

    return loss