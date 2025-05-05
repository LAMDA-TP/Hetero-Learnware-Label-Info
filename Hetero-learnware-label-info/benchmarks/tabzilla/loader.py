import os
import re
import json
import gzip
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# from .utils import get_datasets_meta_data


class TabularDataset(object):
    def __init__(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        cat_idx: list,
        target_type: str,
        num_classes: int,
        num_features: Optional[int] = None,
        num_instances: Optional[int] = None,
        cat_dims: Optional[list] = None,
        split_indeces: Optional[list] = None,
        split_source: Optional[str] = None,
    ) -> None:
        """
        name: name of the dataset
        X: matrix of shape (num_instances x num_features)
        y: array of length (num_instances)
        cat_idx: indices of categorical features
        target_type: {"regression", "classification", "binary"}
        num_classes: 1 for regression 2 for binary, and >2 for classification
        num_features: number of features
        num_instances: number of instances
        split_indeces: specifies dataset splits as a list of dictionaries, with entries "train", "val", and "test".
            each entry specifies the indeces corresponding to the train, validation, and test set.

        ref: https://github.com/naszilla/tabzilla/blob/main/TabZilla/tabzilla_datasets.py
        """
        assert isinstance(X, np.ndarray), "X must be an instance of np.ndarray"
        assert isinstance(y, np.ndarray), "y must be an instance of np.ndarray"
        assert (
            X.shape[0] == y.shape[0]
        ), "X and y must match along their 0-th dimensions"
        assert len(X.shape) == 2, "X must be 2-dimensional"
        assert len(y.shape) == 1, "y must be 1-dimensional"

        if num_instances is not None:
            assert (
                X.shape[0] == num_instances
            ), f"first dimension of X must be equal to num_instances. X has shape {X.shape}"
            assert y.shape == (
                num_instances,
            ), f"shape of y must be (num_instances, ). y has shape {y.shape} and num_instances={num_instances}"
        else:
            num_instances = X.shape[0]

        if num_features is not None:
            assert (
                X.shape[1] == num_features
            ), f"second dimension of X must be equal to num_features. X has shape {X.shape}"
        else:
            num_features = X.shape[1]

        if len(cat_idx) > 0:
            assert (
                max(cat_idx) <= num_features - 1
            ), f"max index in cat_idx is {max(cat_idx)}, but num_features is {num_features}"
        assert target_type in ["regression", "classification", "binary"]

        if target_type == "regression":
            assert num_classes == 1
        elif target_type == "binary":
            assert num_classes == 1
        elif target_type == "classification":
            assert num_classes > 2

        self.name = name
        self.X = X
        self.y = y
        self.cat_idx = cat_idx
        self.target_type = target_type
        self.num_classes = num_classes
        self.num_features = num_features
        self.cat_dims = cat_dims
        self.num_instances = num_instances
        self.split_indeces = split_indeces
        self.split_source = split_source

        pass

    def target_encode(self):
        # print("target_encode...")
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)

        # Sanity check
        if self.target_type == "classification":
            assert self.num_classes == len(
                le.classes_
            ), "num_classes was set incorrectly."

    def cat_feature_encode(self):
        # print("cat_feature_encode...")
        if not self.cat_dims is None:
            raise RuntimeError(
                "cat_dims is already set. Categorical feature encoding might be running twice."
            )
        self.cat_dims = []

        # Preprocess data
        for i in range(self.num_features):
            if self.cat_idx and i in self.cat_idx:
                le = LabelEncoder()
                self.X[:, i] = le.fit_transform(self.X[:, i])

                # Setting this?
                self.cat_dims.append(len(le.classes_))

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "cat_idx": self.cat_idx,
            "cat_dims": self.cat_dims,
            "target_type": self.target_type,
            "num_classes": self.num_classes,
            "num_features": self.num_features,
            "num_instances": self.num_instances,
            "split_source": self.split_source,
        }

    @classmethod
    def read(cls, p: Path):
        """read a dataset from a folder"""

        # make sure that all required files exist in the directory
        X_path = p.joinpath("X.npy.gz")
        y_path = p.joinpath("y.npy.gz")
        metadata_path = p.joinpath("metadata.json")
        split_indeces_path = p / "split_indeces.npy.gz"

        assert X_path.exists(), f"path to X does not exist: {X_path}"
        assert y_path.exists(), f"path to y does not exist: {y_path}"
        assert (
            metadata_path.exists()
        ), f"path to metadata does not exist: {metadata_path}"
        assert (
            split_indeces_path.exists()
        ), f"path to split indeces does not exist: {split_indeces_path}"

        # read data
        with gzip.GzipFile(X_path, "r") as f:
            X = np.load(f, allow_pickle=True)
        with gzip.GzipFile(y_path, "r") as f:
            y = np.load(f)
        with gzip.GzipFile(split_indeces_path, "rb") as f:
            split_indeces = np.load(f, allow_pickle=True)

        # read metadata
        with open(metadata_path, "r") as f:
            kwargs = json.load(f)

        kwargs["X"], kwargs["y"], kwargs["split_indeces"] = X, y, split_indeces
        return cls(**kwargs)

    def write(self, p: Path, overwrite=False) -> None:
        """write the dataset to a new folder. this folder cannot already exist"""

        if not overwrite:
            assert ~p.exists(), f"the path {p} already exists."

        # create the folder
        p.mkdir(parents=True, exist_ok=overwrite)

        # write data
        with gzip.GzipFile(p.joinpath("X.npy.gz"), "w") as f:
            np.save(f, self.X)
        with gzip.GzipFile(p.joinpath("y.npy.gz"), "w") as f:
            np.save(f, self.y)
        with gzip.GzipFile(p.joinpath("split_indeces.npy.gz"), "wb") as f:
            np.save(f, self.split_indeces)

        # write metadata
        with open(p.joinpath("metadata.json"), "w") as f:
            metadata = self.get_metadata()
            json.dump(self.get_metadata(), f, indent=4)


class TabzillaLoader:
    def __init__(self, datasets_path: Optional[str] = None, regenerate_meta_info=False):
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        self.datasets_path = (
            os.path.join(self.root_path, "datasets")
            if datasets_path is None
            else datasets_path
        )
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
        datasets_path = os.path.join(current_folder_path, "datasets")

    def get_dataset(self, dataset_name: str) -> TabularDataset:
        return TabularDataset.read(Path(os.path.join(self.datasets_path, dataset_name)))

    def get_standard_dataset(self, dataset_name: str):
        dataset = self.get_dataset(dataset_name)
        X, y, split_indeces = dataset.X, dataset.y, dataset.split_indeces
        task_type = "classification" if dataset.target_type in ["binary", "classification"] else "regression"
        return {
            "task_name": dataset_name,
            "X": X.astype(np.float32),
            "y": y.astype(np.float32),
            "split_indeces": split_indeces,
            "task_type": task_type,
        }

    def get_split_dataset(self, dataset_name: str):
        """
        Retrieves a split dataset for a specified task and normalizes the target variable for regression.

        This method extracts a dataset by its name, retrieves the corresponding splits for training,
        validation, and testing, and prepares the data accordingly. If the task type is regression,
        it additionally performs min-max normalization on the target variable `y`, scaling it to the
        range [0, 1].

        Parameters
        ----------
        dataset_name : str
            The name of the dataset to retrieve. This name is used to fetch the dataset details,
            including feature matrices `X`, target variables `y`, and indices for data splitting.

        Returns
        -------
        dict
            A dictionary containing detailed structured data related to the task, including:
            - "task_name" : str
                The name of the dataset.
            - "task_type" : str
                The type of machine learning task, either "classification" or "regression".
            - "X" : numpy.ndarray
                The full feature matrix for the dataset.
            - "y" : numpy.ndarray
                The full target variable array, min-max normalized for regression tasks.
            - "X_train" : numpy.ndarray
                The training subset of the feature matrix.
            - "X_test" : numpy.ndarray
                The combined validation and test subset of the feature matrix.
            - "y_train" : numpy.ndarray
                The training subset of the target variable, normalized if the task is regression.
            - "y_test" : numpy.ndarray
                The test subset of the target variable, normalized if the task is regression.

        Notes
        -----
        - The target variable `y` is only normalized for regression tasks; for classification tasks,
          it is used as provided.
        - The validation and test indices are merged to form the test set in this setup.
        """
        dataset_info = self.get_standard_dataset(dataset_name)
        X, y, split_indeces, task_type = (
            dataset_info["X"],
            dataset_info["y"],
            dataset_info["split_indeces"],
            dataset_info["task_type"],
        )
        train_split = split_indeces[0]["train"]
        val_split = split_indeces[0]["val"]
        test_split = split_indeces[0]["test"]
        test_split = np.concatenate([val_split, test_split], axis=0)
        X_train, X_test, y_train, y_test = (
            X[train_split],
            X[test_split],
            y[train_split],
            y[test_split],
        )
        if task_type == "regression":
            # use min max scaler for regression y
            y_scaler = MinMaxScaler()
            y_scaler.fit(y_train.reshape(-1, 1))
            y_train = y_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
        return {
            "task_name": dataset_name,
            "task_type": task_type,  # "classification" or "regression
            "X": X,
            "y": y,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }