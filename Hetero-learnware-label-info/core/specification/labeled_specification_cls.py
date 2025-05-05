import torch
import numpy as np

from learnware.specification import RegularStatSpecification
from learnware.utils import allocate_cuda_idx, choose_device
from learnware.specification.regular.table.rkme import torch_rbf_kernel, rkme_solve_qp
from typing import Union, List
from loguru import logger
from scipy.linalg import block_diag as scipy_block_diag
import json
import codecs
import os


class LabeledSpecificationCLS(RegularStatSpecification):

    def __init__(
        self, gamma: float = 0.1, cuda_idx: int = None, weights_conditioned: float = 1
    ):
        """Initializing LabeledSpecificationCLS parameters.

        Parameters
        ----------
        gamma : float
            Bandwidth in gaussian kernel, by default 0.1.
        cuda_idx : int
            A flag indicating whether use CUDA during RKME computation. -1 indicates CUDA not used. None indicates automatically choose device
        """
        self.z = None
        self.beta = None
        self.y = None
        self.gamma = gamma
        self.weights_conditioned = weights_conditioned
        self.num_points = 0
        self._cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        torch.cuda.empty_cache()
        self._device = choose_device(cuda_idx=self._cuda_idx)
        super(LabeledSpecificationCLS, self).__init__(type=self.__class__.__name__)

    @property
    def device(self):
        return self._device

    def get_beta(self) -> np.ndarray:
        """Move beta(RKME weights) back to memory accessible to the CPU.

        Returns
        -------
        np.ndarray
            A copy of beta in CPU memory.
        """
        return self.beta.detach().cpu().numpy()

    def get_z(self) -> np.ndarray:
        """Move z(RKME reduced set points) back to memory accessible to the CPU.

        Returns
        -------
        np.ndarray
            A copy of z in CPU memory.
        """
        return self.z.detach().cpu().numpy()

    def get_y(self) -> np.ndarray:
        """Move y(RKME labels) back to memory accessible to the CPU.

        Returns
        -------
        np.ndarray
            A copy of y in CPU memory.
        """
        return self.y.detach().cpu().numpy()

    @staticmethod
    def clean_and_fill_data(X):
        """
        Clean the array X by replacing all infinite values with NaN, then fill any NaN values in each column with the mean of that column.
        Raises ValueError if a column is entirely NaN or infinite.

        Parameters:
            X (np.ndarray): A 2D numpy array to be cleaned and filled.

        Returns:
            np.ndarray: The cleaned and filled numpy array.
        """
        # Replace inf and -inf with NaN
        X[np.isinf(X) | np.isneginf(X) | np.isposinf(X) | np.isneginf(X)] = np.nan

        # Check for NaN values in the array
        if np.any(np.isnan(X)):
            for col in range(X.shape[1]):
                is_nan = np.isnan(X[:, col])
                if np.any(is_nan):
                    if np.all(is_nan):
                        # Raise an error if all values in a column are NaN or inf
                        raise ValueError(
                            f"All values in column {col} are exceptional, e.g., NaN and Inf."
                        )
                    # Fill NaN with column mean, excluding NaN values from the mean calculation
                    col_mean = np.nanmean(X[:, col])
                    X[:, col] = np.where(is_nan, col_mean, X[:, col])

        return X

    def generate_stat_spec_from_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        K: int = 100,
        step_size: float = 0.1,
        steps: int = 10,
        nonnegative_beta: bool = True,
        reduce: bool = True,
    ):
        """Construct reduced set from raw dataset using iterative optimization.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format.
        K : int
            Size of the construced reduced set.
        step_size : float
            Step size for gradient descent in the iterative optimization.
        steps : int
            Total rounds in the iterative optimization.
        nonnegative_beta : bool, optional
            True if weights for the reduced set are intended to be kept non-negative, by default False.
        reduce : bool, optional
            Whether shrink original data to a smaller set, by default True
        """
        alpha = None
        self.num_points = X.shape[0]
        if K >= self.num_points:
            K_original = K
            K = max(1, self.num_points * 2 // 3)
            logger.warning(f"K is too large, reset K from {K_original} to {K}.")
        else:
            K = K

        X_shape = X.shape
        Z_shape = tuple([K] + list(X_shape)[1:])
        X = X.reshape(self.num_points, -1)
        X, y = self.reorder_samples(X, y)

        # Check data values
        X = self.clean_and_fill_data(X)

        if not reduce:
            self.z = X.reshape(X_shape)
            self.beta = 1 / self.num_points * np.ones(self.num_points)
            self.y = y
            self.z = torch.from_numpy(self.z).double().to(self._device)
            self.beta = torch.from_numpy(self.beta).double().to(self._device)
            self.y = torch.from_numpy(self.y).to(self._device)
            return

        # Initialize Z by clustering, utiliing kmeans to speed up the process.
        self._init_z_y_by_kmeans(X, y, K)
        self._update_beta(X, y, nonnegative_beta)

        # Alternating optimize Z and beta
        for i in range(steps):
            self._update_z(alpha, X, y, step_size)
            self._update_beta(X, y, nonnegative_beta)

        # Reshape to original dimensions
        self.z = self.z.reshape(Z_shape)

    def generate_stat_spec_from_ssl_data(
        self,
        X_unlabel: np.ndarray,  # include the X_label
        X_label: np.ndarray,
        y_label: np.ndarray,
        K: int = 100,
        step_size: float = 0.1,
        steps: int = 10,
        nonnegative_beta: bool = True,
        reduce: bool = True,
    ):
        alpha_1 = None
        alpha_2 = None
        self.num_points = X_unlabel.shape[0]
        self.num_points_label = X_label.shape[0]
        if K >= self.num_points:
            K_original = K
            K = max(1, self.num_points * 2 // 3)
            logger.warning(f"K is too large, reset K from {K_original} to {K}.")
        else:
            K = K

        X_unlabel_shape = X_unlabel.shape
        X_label_shape = X_label.shape
        Z_shape = tuple([K] + list(X_unlabel_shape)[1:])
        X_unlabel = X_unlabel.reshape(self.num_points, -1)
        X_label = X_label.reshape(self.num_points_label, -1)
        X_label, y_label = self.reorder_samples(X_label, y_label)

        # check data values
        X_label = self.clean_and_fill_data(X_label)
        X_unlabel = self.clean_and_fill_data(X_unlabel)

        if not reduce:
            self.z = X_label.reshape(X_label_shape)
            self.beta = 1 / self.num_points_label * np.ones(self.num_points_label)
            self.y = y_label
            self.z = torch.from_numpy(self.z).double().to(self._device)
            self.beta = torch.from_numpy(self.beta).double().to(self._device)
            self.y = torch.from_numpy(self.y).to(self._device)
            return

        # Initialize Z by clustering, utiliing kmeans to speed up the process.
        self._init_z_y_by_kmeans(X_label, y_label, K)
        self._update_beta_ssl(X_unlabel, X_label, y_label, nonnegative_beta)

        # Alternating optimize Z and beta
        for i in range(steps):
            self._update_z_ssl(alpha_1, alpha_2, X_unlabel, X_label, y_label, step_size)
            self._update_beta_ssl(X_unlabel, X_label, y_label, nonnegative_beta)

        # Reshape to original dimensions
        self.z = self.z.reshape(Z_shape)

    @staticmethod
    def reorder_samples(X, y):
        """
        Reorder samples in X based on the values of y.

        Parameters
        ----------
        X : np.ndarray
            The input feature array.
        y : np.ndarray
            The target labels array, which should contain integer labels.

        Returns
        -------
        np.ndarray
            The rearranged array of features where samples are ordered
            by the ascending order of labels in y.

        Raises
        ------
        ValueError
            If X and y do not have the same length.

        """
        # Ensure X and y have the same length on the first dimension
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        # Compute indices for sorting based on labels in y
        sorted_indices = np.argsort(y)

        # Use the sorted indices to reorder X and y
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]

        return X_sorted, y_sorted

    def _init_z_y_by_kmeans(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        K: int,
    ):
        if isinstance(X, np.ndarray):
            X = X.astype("float32")
            X = torch.from_numpy(X)
        if isinstance(y, np.ndarray):
            y = y.astype("float32")
            y = torch.from_numpy(y)

        X = X.to(self._device)
        y = y.to(self._device)

        try:
            from fast_pytorch_kmeans import KMeans
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "RKMETableSpecification is not available because 'fast_pytorch_kmeans' is not installed! Please install it manually."
            )

        # Determine the class ratios and the number of samples per class in the new dataset
        unique_classes, counts = torch.unique(y, return_counts=True)
        class_ratios = counts.float() / counts.sum()
        samples_per_class = (class_ratios * K).round().int()

        # Adjust samples_per_class to ensure the sum equals K
        discrepancy = K - samples_per_class.sum()
        if discrepancy != 0:
            max_count_idx = torch.argmax(counts)
            samples_per_class[max_count_idx] += discrepancy

        centroids_per_class = []
        labels_per_class = []

        # Perform K-means within each class
        for cls, num_samples in zip(unique_classes, samples_per_class):
            class_mask = y == cls
            class_samples = X[class_mask]
            if class_samples.shape[0] == 0 or num_samples <= 0:
                logger.warning(
                    f"No samples or negative sample count for class {cls.item()}. Adjusted samples count: {num_samples.item()}. Skipping this class."
                )
                continue

            # Initialize KMeans
            kmeans = KMeans(
                n_clusters=num_samples.item(), mode="euclidean", max_iter=100, verbose=0
            )
            kmeans.fit(class_samples)
            centroids_per_class.append(kmeans.centroids)
            labels_per_class.extend([cls] * num_samples.item())

        # Concatenate centroids from each class to form the final set of centroids
        self.z = torch.cat(centroids_per_class, dim=0).double()
        self.y = torch.tensor(labels_per_class, device=self._device)

    def _update_beta(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        nonnegative_beta: bool = True,
    ):
        """Fix Z and update beta using its closed-form solution.

        Parameters
        ----------
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        nonnegative_beta : bool, optional
            True if weights for the reduced set are intended to be kept non-negative, by default False.
        """
        Z = self.z
        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self._device).double()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self._device).double()

        K = torch_rbf_kernel(Z, Z, gamma=self.gamma).to(self._device)
        Z_len_list = self._get_class_num_list(self.y)
        K_block = self._generate_block_diagonal_matrix(
            K=K, col_len_list=Z_len_list, row_len_list=Z_len_list
        )
        K += K_block * self.weights_conditioned

        C = torch_rbf_kernel(Z, X, gamma=self.gamma).to(self._device)
        X_len_list = self._get_class_num_list(y)
        C_block = self._generate_block_diagonal_matrix(
            K=C, col_len_list=X_len_list, row_len_list=Z_len_list
        )
        C += C_block * self.weights_conditioned
        C = torch.sum(C, dim=1) / X.shape[0]

        if nonnegative_beta:
            beta, _ = rkme_solve_qp(K, C)
            beta = beta.to(self._device)
        else:
            beta = (
                torch.linalg.inv(K + torch.eye(K.shape[0]).to(self._device) * 1e-5) @ C
            )

        self.beta = beta

    def _update_beta_ssl(
        self,
        X_unlabel: Union[np.ndarray, torch.Tensor],
        X_label: Union[np.ndarray, torch.Tensor],
        y_label: Union[np.ndarray, torch.Tensor],
        nonnegative_beta: bool = True,
    ):
        Z = self.z
        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self._device).double()

        if not torch.is_tensor(X_unlabel):
            X_unlabel = torch.from_numpy(X_unlabel)
        X_unlabel = X_unlabel.to(self._device).double()

        if not torch.is_tensor(X_label):
            X_label = torch.from_numpy(X_label)
        X_label = X_label.to(self._device).double()

        K = torch_rbf_kernel(Z, Z, gamma=self.gamma).to(self._device)
        Z_len_list = self._get_class_num_list(self.y)
        K_block = self._generate_block_diagonal_matrix(
            K=K, col_len_list=Z_len_list, row_len_list=Z_len_list
        )
        K += K_block * self.weights_conditioned

        C1 = torch_rbf_kernel(Z, X_unlabel, gamma=self.gamma).to(self._device)
        C1 = torch.sum(C1, dim=1) / X_unlabel.shape[0]
        C2 = torch_rbf_kernel(Z, X_label, gamma=self.gamma).to(self._device)
        X_len_list = self._get_class_num_list(y_label)
        C2_block = self._generate_block_diagonal_matrix(
            K=C2, col_len_list=X_len_list, row_len_list=Z_len_list
        )
        C2 += C2_block * self.weights_conditioned
        C2 = torch.sum(C2, dim=1) / X_label.shape[0]
        C = C1 + C2

        if nonnegative_beta:
            beta, _ = rkme_solve_qp(K, C)
            beta = beta.to(self._device)
        else:
            beta = (
                torch.linalg.inv(K + torch.eye(K.shape[0]).to(self._device) * 1e-5) @ C
            )

        self.beta = beta

    @staticmethod
    def _generate_block_diagonal_matrix(
        K: Union[np.ndarray, torch.Tensor],
        col_len_list: List[int],
        row_len_list: List[int],
    ):
        # Check if the sum of col_len_list and row_len_list matches the dimensions of K
        if sum(col_len_list) != K.shape[1] or sum(row_len_list) != K.shape[0]:
            raise ValueError(
                "The sum of col_len_list or row_len_list does not match the dimensions of K"
            )

        # Initialize a list to store block matrices
        blocks = []

        # Define starting indices for rows and columns
        row_start = 0
        col_start = 0

        # Extract block matrices along the diagonal
        for col_len, row_len in zip(col_len_list, row_len_list):
            # Extract the block
            block = K[row_start : row_start + row_len, col_start : col_start + col_len]
            blocks.append(block)
            # Update the starting indices
            row_start += row_len
            col_start += col_len

        # Create a block diagonal matrix from the list of blocks
        if isinstance(K, np.ndarray):
            # Using numpy's block_diag function to create block diagonal matrix
            block_diag_matrix = scipy_block_diag(*blocks)
        elif isinstance(K, torch.Tensor):
            # Using torch's block_diag function to create block diagonal matrix
            block_diag_matrix = torch.block_diag(*blocks)

        return block_diag_matrix

    @staticmethod
    def _get_class_num_list(y: Union[np.ndarray, torch.Tensor]) -> List[int]:
        """Get the number of samples for each class in the input labels.

        Parameters
        ----------
        y : Union[np.ndarray, torch.Tensor]
            Input labels.

        Returns
        -------
        List[int]
            Number of samples for each class in the input labels.

        Raises
        ------
        ValueError
            If the input labels are not in the expected format.

        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if not torch.is_tensor(y):
            raise ValueError("Input labels must be a numpy array or a PyTorch tensor.")

        # Get the unique classes and their counts
        unique_classes, counts = torch.unique(y, return_counts=True)

        # Convert the counts to a list
        class_num_list = counts.tolist()

        return class_num_list

    def _update_z(
        self,
        alpha: float,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        step_size: float,
    ):
        """Fix beta and update Z using gradient descent.

        Parameters
        ----------
        alpha : int
            Normalization factor.
        X : np.ndarray or torch.tensor
            Raw data in np.ndarray format or torch.tensor format.
        step_size : float
            Step size for gradient descent.
        """
        gamma = self.gamma
        Z = self.z
        beta = self.beta
        Z_y = self.y

        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self._device).double()

        if not torch.is_tensor(beta):
            beta = torch.from_numpy(beta)
        beta = beta.to(self._device).double()

        if not torch.is_tensor(X):
            X = torch.from_numpy(X)
        X = X.to(self._device).double()

        if not torch.is_tensor(y):
            y = torch.from_numpy(y)
        y = y.to(self._device)

        grad_Z = torch.zeros_like(Z)

        for i in range(Z.shape[0]):
            z_i = Z[i, :].reshape(1, -1)
            z_i_label = Z_y[i]

            # Create match vector for Z and z_i
            match_Z = (Z_y == z_i_label).double() * self.weights_conditioned + 1.0
            # Create match vector for X and z_i
            match_X = (y == z_i_label).double() * self.weights_conditioned + 1.0

            term_1 = 2 * (beta * match_Z * torch_rbf_kernel(z_i, Z, gamma)) @ (z_i - Z)
            if alpha is not None:
                term_2 = (
                    -2 * (alpha * match_X * torch_rbf_kernel(z_i, X, gamma)) @ (z_i - X)
                )
            else:
                term_2 = (
                    -2
                    * (match_X * torch_rbf_kernel(z_i, X, gamma) / self.num_points)
                    @ (z_i - X)
                )
            grad_Z[i, :] = -2 * gamma * beta[i] * (term_1 + term_2)

        Z = Z - step_size * grad_Z
        self.z = Z

    def _update_z_ssl(
        self,
        alpha_1: float,
        alpha_2: float,
        X_unlabel: Union[np.ndarray, torch.Tensor],
        X_label: Union[np.ndarray, torch.Tensor],
        y_label: Union[np.ndarray, torch.Tensor],
        step_size: float,
    ):
        gamma = self.gamma
        Z = self.z
        beta = self.beta
        Z_y = self.y

        if not torch.is_tensor(Z):
            Z = torch.from_numpy(Z)
        Z = Z.to(self._device).double()

        if not torch.is_tensor(beta):
            beta = torch.from_numpy(beta)
        beta = beta.to(self._device).double()

        if not torch.is_tensor(X_unlabel):
            X_unlabel = torch.from_numpy(X_unlabel)
        X_unlabel = X_unlabel.to(self._device).double()

        if not torch.is_tensor(X_label):
            X_label = torch.from_numpy(X_label)
        X_label = X_label.to(self._device).double()

        if not torch.is_tensor(y_label):
            y_label = torch.from_numpy(y_label)
        y_label = y_label.to(self._device)

        grad_Z = torch.zeros_like(Z)

        for i in range(Z.shape[0]):
            z_i = Z[i, :].reshape(1, -1)
            z_i_label = Z_y[i]

            # Create match vector for Z and z_i
            match_Z = (Z_y == z_i_label).double() * self.weights_conditioned + 1.0
            # Create match vector for X_label and z_i
            match_X_label = (y_label == z_i_label).double() * self.weights_conditioned

            term_1 = 2 * (beta * match_Z * torch_rbf_kernel(z_i, Z, gamma)) @ (z_i - Z)
            if alpha_1 is not None:
                term_2 = (
                    -2
                    * (alpha_1 * torch_rbf_kernel(z_i, X_unlabel, gamma))
                    @ (z_i - X_unlabel)
                )
            else:
                term_2 = (
                    -2
                    * (torch_rbf_kernel(z_i, X_unlabel, gamma) / self.num_points)
                    @ (z_i - X_unlabel)
                )
            if alpha_2 is not None:
                term_3 = (
                    -2
                    * (alpha_2 * match_X_label * torch_rbf_kernel(z_i, X_label, gamma))
                    @ (z_i - X_label)
                )
            else:
                term_3 = (
                    -2
                    * (torch_rbf_kernel(z_i, X_label, gamma) / self.num_points_label)
                    @ (z_i - X_label)
                )
            grad_Z[i, :] = -2 * gamma * beta[i] * (term_1 + term_2 + term_3)

        Z = Z - step_size * grad_Z
        self.z = Z

    def save(self, filepath: str):
        """Save the computed RKME specification to a specified path in JSON format.

        Parameters
        ----------
        filepath : str
            The specified saving path.
        """
        save_path = filepath
        rkme_to_save = self.get_states()
        if torch.is_tensor(rkme_to_save["z"]):
            rkme_to_save["z"] = rkme_to_save["z"].detach().cpu().numpy()
        rkme_to_save["z"] = rkme_to_save["z"].tolist()
        if torch.is_tensor(rkme_to_save["beta"]):
            rkme_to_save["beta"] = rkme_to_save["beta"].detach().cpu().numpy()
        rkme_to_save["beta"] = rkme_to_save["beta"].tolist()
        if torch.is_tensor(rkme_to_save["y"]):
            rkme_to_save["y"] = rkme_to_save["y"].detach().cpu().numpy()
        rkme_to_save["y"] = rkme_to_save["y"].tolist()
        with codecs.open(save_path, "w", encoding="utf-8") as fout:
            json.dump(rkme_to_save, fout, separators=(",", ":"))

    def load(self, filepath: str) -> bool:
        """Load a RKME specification file in JSON format from the specified path.

        Parameters
        ----------
        filepath : str
            The specified loading path.

        Returns
        -------
        bool
            True if the RKME is loaded successfully.
        """
        # Load JSON file:
        load_path = filepath
        if os.path.exists(load_path):
            with codecs.open(load_path, "r", encoding="utf-8") as fin:
                obj_text = fin.read()
            rkme_load = json.loads(obj_text)
            rkme_load["z"] = torch.from_numpy(np.array(rkme_load["z"]))
            rkme_load["beta"] = torch.from_numpy(np.array(rkme_load["beta"]))
            rkme_load["y"] = torch.from_numpy(np.array(rkme_load["y"]))

            for d in self.get_states():
                if d in rkme_load.keys():
                    if d == "type" and rkme_load[d] != self.type:
                        raise TypeError(
                            f"The type of loaded RKME ({rkme_load[d]}) is different from the expected type ({self.type})!"
                        )
                    setattr(self, d, rkme_load[d])