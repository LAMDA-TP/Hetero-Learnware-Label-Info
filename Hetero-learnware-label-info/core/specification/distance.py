# python -m Hetero_learnware_main.core.specification.distance

import torch
from .labeled_specification import (
    LabeledSpecification,
    SystemSpecification,
    LabeledRKMESpecification,
)
from learnware.specification import RKMETableSpecification
from learnware.utils import allocate_cuda_idx, choose_device
from typing import Union


class MMDDistance:

    def __init__(
        self,
        task_type: str,
        gamma: float,
        cond_mmd_coef: float = 1,
        num_bins: int = None,
        cuda_idx: int = None,
    ):
        self.task_type = task_type
        self.cond_mmd_coef = cond_mmd_coef
        self.num_bins = num_bins
        self.gamma = gamma

        if task_type not in ["classification", "regression"]:
            raise ValueError(
                f"Task type {task_type} not supported. Use 'classification' or 'regression'"
            )
        if task_type == "regression" and num_bins is None:
            raise ValueError("num_bins must be provided for regression tasks")

        self.cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        self.device = choose_device(cuda_idx=self.cuda_idx)

    def _check_same_device(self, *args):
        """
        Check if all the input tensors are on the same device.

        Parameters
        ----------
        args : list
            List of tensors to check.

        Returns
        -------
        bool
            True if all tensors are on the same device, False otherwise.
        """
        devices = [arg.device for arg in args]
        if all(device == devices[0] for device in devices) is False:
            raise ValueError("All input tensors must be on the same device.")

    def _to_same_device(self, *args):
        """
        Move all the input tensors to the same device.

        Parameters
        ----------
        args : list
            List of tensors to move to the same device.

        Returns
        -------
        list
            List of tensors on the same device.
        """
        device = self.device
        return [arg.to(device) for arg in args]

    def _to_default_device(
        self,
        spec: Union[
            SystemSpecification,
            LabeledSpecification,
            RKMETableSpecification,
            LabeledRKMESpecification,
        ],
    ):
        """
        Move the specification to the default device.

        Parameters
        ----------
        spec : SystemSpecification
            The specification to move to the default device.

        Returns
        -------
        SystemSpecification
            The specification on the default device.
        """
        if isinstance(spec, SystemSpecification):
            spec.x = spec.x.to(self.device)
            spec.weights = spec.weights.to(self.device)
            spec.y = spec.y.to(self.device)
        elif isinstance(spec, RKMETableSpecification):
            spec.z = spec.z.to(self.device)
            spec.beta = spec.beta.to(self.device)
        elif isinstance(spec, LabeledSpecification):
            spec.rkme.z = spec.rkme.z.to(self.device)
            spec.rkme.beta = spec.rkme.beta.to(self.device)
            spec.y = spec.y.to(self.device)
        elif isinstance(spec, LabeledRKMESpecification):
            spec.rkme.z = spec.rkme.z.to(self.device)
            spec.rkme.beta = spec.rkme.beta.to(self.device)
            spec.y = spec.y.to(self.device)

    def _gaussian_kernel(self, x1: torch.Tensor, x2: torch.Tensor):
        """
        Compute the Gaussian kernel between two sets of samples.

        Parameters
        ----------
        x1 : torch.Tensor
            First set of samples.
        x2 : torch.Tensor
            Second set of samples.

        Returns
        -------
        torch.Tensor
            The computed Gaussian kernel matrix.
        """
        x1 = x1.double()
        x2 = x2.double()
        X12norm = (
            torch.sum(x1**2, 1, keepdim=True)
            - 2 * x1 @ x2.T
            + torch.sum(x2**2, 1, keepdim=True).T
        )
        return torch.exp(-X12norm * self.gamma)

    def _compute_mmd(
        self,
        user_X: torch.Tensor,
        user_weight: torch.Tensor,
        target_X: torch.Tensor,
        target_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Maximum Mean Discrepancy (MMD) between the user and target datasets.

        Parameters
        ----------
        user_X : torch.Tensor
            Transformed user data.
        user_weight : torch.Tensor
            Weights of the user data.
        target_X : torch.Tensor
            Target data.
        target_weight : torch.Tensor
            Weights of the target data.

        Returns
        -------
        torch.Tensor
            The computed MMD loss.
        """
        self._check_same_device(user_X, user_weight, target_X, target_weight)

        term1 = torch.sum(
            self._gaussian_kernel(user_X, user_X) * (user_weight.T @ user_weight)
        )
        term2 = torch.sum(
            self._gaussian_kernel(user_X, target_X) * (user_weight.T @ target_weight)
        )
        term3 = torch.sum(
            self._gaussian_kernel(target_X, target_X)
            * (target_weight.T @ target_weight)
        )
        return term1 - 2 * term2 + term3

    def _compute_conditioned_mmd(
        self,
        user_X: torch.Tensor,
        user_weight: torch.Tensor,
        user_y: torch.Tensor,
        target_X: torch.Tensor,
        target_weight: torch.Tensor,
        target_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the conditioned MMD for classification or regression tasks.
        For classification, groups are formed based on class labels.
        For regression, data is binned into quantiles.

        Parameters
        ----------
        user_X : torch.Tensor
            Transformed user data.
        user_weight : torch.Tensor
            Weights of the user data.
        user_y : torch.Tensor
            Labels or continuous values of the user data.
        target_X : torch.Tensor
            Target data.
        target_weight : torch.Tensor
            Weights of the target data.
        target_y : torch.Tensor
            Labels or continuous values of the target data.
        task_type : str, optional
            Task type, either 'classification' or 'regression'.

        Returns
        -------
        torch.Tensor
            The computed conditioned MMD loss.
        """
        self._check_same_device(
            user_X, user_weight, user_y, target_X, target_weight, target_y
        )

        if self.task_type == "classification":
            labels = torch.unique(torch.cat([user_y, target_y]))
        elif self.task_type == "regression" and self.num_bins is not None:
            # Combine all values and compute quantile edges
            all_values = torch.cat([user_y, target_y])
            quantiles = torch.quantile(
                all_values,
                torch.linspace(0, 1, self.num_bins + 1)
                .to(all_values.dtype)
                .to(self.device),
            )
            labels = (
                quantiles[:-1] + quantiles[1:]
            ) / 2  # Compute midpoints for bin labels
        else:
            raise ValueError(
                "Invalid task type or num_bins not provided for regression."
            )

        total_mmd = 0.0

        if self.task_type == "classification":
            label_indices = [user_y == label for label in labels]
            target_label_indices = [target_y == label for label in labels]
        elif self.task_type == "regression":
            label_indices = [
                (user_y >= quantiles[i]) & (user_y < quantiles[i + 1])
                for i in range(len(quantiles) - 1)
            ]
            target_label_indices = [
                (target_y >= quantiles[i]) & (target_y < quantiles[i + 1])
                for i in range(len(quantiles) - 1)
            ]

        for user_indices, target_indices in zip(label_indices, target_label_indices):
            user_X_label = user_X[user_indices]
            user_weight_label = user_weight[:, user_indices]
            target_X_label = target_X[target_indices]
            target_weight_label = target_weight[:, target_indices]

            if user_X_label.size(0) > 0 and target_X_label.size(0) > 0:
                mmd_label = self._compute_mmd(
                    user_X_label, user_weight_label, target_X_label, target_weight_label
                )
                total_mmd += mmd_label

        return total_mmd

    def _compute_total_mmd(
        self,
        user_X: torch.Tensor,
        user_weight: torch.Tensor,
        user_y: torch.Tensor,
        target_X: torch.Tensor,
        target_weight: torch.Tensor,
        target_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the total MMD for classification or regression tasks.
        For classification, groups are formed based on class labels.
        For regression, data is binned into quantiles.

        Parameters
        ----------
        user_X : torch.Tensor
            Transformed user data.
        user_weight : torch.Tensor
            Weights of the user data.
        user_y : torch.Tensor
            Labels or continuous values of the user data.
        target_X : torch.Tensor
            Target data.
        target_weight : torch.Tensor
            Weights of the target data.
        target_y : torch.Tensor
            Labels or continuous values of the target data.
        task_type : str, optional
            Task type, either 'classification' or 'regression'.

        Returns
        -------
        torch.Tensor
            The computed total MMD loss.
        """
        return self._compute_mmd(
            user_X, user_weight, target_X, target_weight
        ) + self.cond_mmd_coef * self._compute_conditioned_mmd(
            user_X,
            user_weight,
            user_y,
            target_X,
            target_weight,
            target_y,
        )

    def _check_gamma(self, spec_1: LabeledSpecification, spec_2: LabeledSpecification):
        gamma_1 = spec_1.gamma
        gamma_2 = spec_2.gamma
        if gamma_1 != gamma_2:
            raise ValueError(
                "The gamma values of the two specifications are different!"
            )
        if gamma_1 != self.gamma or gamma_2 != self.gamma:
            raise ValueError(
                "The gamma value of the specifications does not match the provided mmd gamma value!"
            )

    def _get_paras(
        self,
        spec: Union[
            LabeledSpecification,
            SystemSpecification,
            RKMETableSpecification,
            LabeledRKMESpecification,
        ],
    ):
        if isinstance(spec, SystemSpecification):
            X = torch.Tensor(spec.x)
            weight = torch.Tensor(spec.weights.reshape(1, -1))
            y = torch.Tensor(spec.y)
        elif isinstance(spec, RKMETableSpecification):
            X = spec.z
            weight = spec.beta.reshape(1, -1)
            y = None
        elif isinstance(spec, LabeledSpecification):
            X = spec.rkme.z
            weight = spec.rkme.beta.reshape(1, -1)
            y = torch.Tensor(spec.y)
        elif isinstance(spec, LabeledRKMESpecification):
            X = spec.rkme.z
            weight = spec.rkme.beta.reshape(1, -1)
            y = torch.Tensor(spec.y)
        return X, weight, y

    def mmd(
        self,
        spec_1: Union[
            LabeledSpecification,
            SystemSpecification,
            RKMETableSpecification,
            LabeledRKMESpecification,
        ],
        spec_2: Union[
            LabeledSpecification,
            SystemSpecification,
            RKMETableSpecification,
            LabeledRKMESpecification,
        ],
    ) -> torch.Tensor:
        """
        Compute the Maximum Mean Discrepancy (MMD) between two specifications.

        Parameters
        ----------
        spec_1 : LabeledSpecification
            First specification.
        spec_2 : LabeledSpecification
            Second specification.

        Returns
        -------
        torch.Tensor
            The computed MMD loss.
        """
        self._check_gamma(spec_1, spec_2)

        X1, weight1, y1 = self._get_paras(spec_1)
        X2, weight2, y2 = self._get_paras(spec_2)

        return self._compute_mmd(X1, weight1, X2, weight2)

    def mmd_cond(
        self,
        spec_1: Union[
            LabeledSpecification, SystemSpecification, LabeledRKMESpecification
        ],
        spec_2: Union[
            LabeledSpecification, SystemSpecification, LabeledRKMESpecification
        ],
    ) -> torch.Tensor:
        """
        Compute the conditioned Maximum Mean Discrepancy (MMD) between two specifications.

        Parameters
        ----------
        spec_1 : LabeledSpecification
            First specification.
        spec_2 : LabeledSpecification
            Second specification.

        Returns
        -------
        torch.Tensor
            The computed conditioned MMD loss.
        """
        self._check_gamma(spec_1, spec_2)

        X1, weight1, y1 = self._get_paras(spec_1)
        X2, weight2, y2 = self._get_paras(spec_2)

        return self._compute_conditioned_mmd(
            X1,
            weight1,
            y1,
            X2,
            weight2,
            y2,
        )

    def mmd_total(
        self,
        spec_1: Union[
            LabeledSpecification, SystemSpecification, LabeledRKMESpecification
        ],
        spec_2: Union[
            LabeledSpecification, SystemSpecification, LabeledRKMESpecification
        ],
    ) -> torch.Tensor:
        """
        Compute the total Maximum Mean Discrepancy (MMD) between two specifications.

        Parameters
        ----------
        spec_1 : LabeledSpecification
            First specification.
        spec_2 : LabeledSpecification
            Second specification.

        Returns
        -------
        torch.Tensor
            The computed total MMD loss.
        """
        return self.mmd(spec_1, spec_2) + self.cond_mmd_coef * self.mmd_cond(
            spec_1, spec_2
        )