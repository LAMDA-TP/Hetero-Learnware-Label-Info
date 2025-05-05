import math
import numpy as np

from ..utils import fill_nan_with_exception_early
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger
from sklearn.model_selection import StratifiedShuffleSplit
from itertools import combinations
import pandas as pd

class TaskGenerator:

    def __init__(
        self,
        split_data: dict,
        n_feature_splits: int = 4,
        n_learnware_blocks: int = 3,
        seed: int = 0,
    ):
        if n_learnware_blocks >= n_feature_splits:
            logger.error(f"n_learnware_blocks should be less than n_feature_splits.")
            exit(1)

        self.task_name = split_data["task_name"]
        self.X_train = split_data["X_train"]
        self.y_train = split_data["y_train"]
        self.X_test = split_data["X_test"]
        self.y_test = split_data["y_test"]
        self.task_type = split_data["task_type"]

        self.n_classes = len(set(self.y_train))
        # self.n_class_splits = math.floor(self.n_classes / 2)
        self.n_class_splits = 1
        self.n_features = self.X_train.shape[1]
        self.n_feature_splits = n_feature_splits
        self.n_learnware_blocks = n_learnware_blocks

        self.class_blocks = self._generate_class_blocks()
        self.feature_blocks = self._generate_feature_blocks()
        self.feature_blocks_assignments = self._generate_feature_blocks_assignments()
        self.n_learnware_features = len(self.feature_blocks_assignments)
        self.learnware_feature_split_info = self._get_learnware_feature_split_info()
        self.rng = np.random.default_rng(seed)

    def get_learnware_task_ids(self):
        """
        Generate a list of IDs for all possible learnware tasks based on the number of feature splits and class splits.

        This method calculates the total number of learnware tasks by multiplying the number of learnware features
        (feature splits not including the full set) with the number of class splits, and returns a sequential list of IDs for these tasks.

        Returns
        -------
        list of int
            A list of integers representing all possible task IDs, ranging from 0 to total_learnware_tasks-1.
        """
        total_learnware_tasks = self.n_learnware_features * self.n_class_splits
        return list(range(total_learnware_tasks))

    def get_learnware_task_by_id(self, task_id):
        # Calculate feature block ID and class block ID from task_id
        feature_block_id = task_id // self.n_class_splits
        class_block_id = task_id % self.n_class_splits
        # logger.info(f"feature_block_id: {feature_block_id}, class_block_id: {class_block_id}")

        # Select the feature blocks using the feature_blocks_assignments
        feature_block_list = self.feature_blocks_assignments[feature_block_id]

        # Class block is directly indexed since each is distinct
        class_block_list = [class_block_id]

        # Retrieve the sub-dataset for the given configuration
        return self.get_sub_dataset(feature_block_list, class_block_list)

    def _get_user_task_ids(self, n_mix):
        total_user_tasks = math.comb(self.n_feature_splits, n_mix)
        return list(range(total_user_tasks))

    def get_all_user_task_ids(self):
        """
        Generate a dictionary of all user task IDs, keyed by the number of mixed classes (`n_mix`).

        Iterates over all possible numbers of class combinations, from 1 up to the number of class splits,
        generating task IDs for each combination level using `_get_user_task_ids`.

        Returns
        -------
        dict
            A dictionary where each key is an integer `n_mix` representing the number of classes combined,
            and each value is a list of task IDs for that combination level.
        """
        all_user_task_ids = {}
        for i in range(1, self.n_feature_splits - 1):
            all_user_task_ids[i] = self._get_user_task_ids(i)
        return all_user_task_ids

    def get_user_task_by_id(self, task_id, n_mix):
        # Generate all combinations of class block indices based on n_mix

        all_combinations = list(combinations(range(self.n_feature_splits), n_mix))
        feature_block_list = all_combinations[task_id]

        # Retrieve the sub-dataset for the given combination of class blocks
        return self.get_sub_dataset(feature_block_list, [0]), feature_block_list

    def _generate_class_blocks(self):
        class_blocks = []
        classes_per_block = self.n_classes // self.n_class_splits
        remainder_classes = self.n_classes % self.n_class_splits

        start = 0
        for i in range(self.n_class_splits):
            if remainder_classes > 0:
                end = start + classes_per_block + 1
                remainder_classes -= 1
            else:
                end = start + classes_per_block
            class_blocks.append(np.arange(start, end))
            start = end

        return class_blocks

    def _generate_feature_blocks(self):
        feature_blocks = []
        features_per_block = self.n_features // self.n_feature_splits
        remainder_features = self.n_features % self.n_feature_splits

        start = 0
        for i in range(self.n_feature_splits):
            if remainder_features > 0:
                end = start + features_per_block + 1
                remainder_features -= 1
            else:
                end = start + features_per_block
            feature_blocks.append(np.arange(start, end))
            start = end

        return feature_blocks

    def _get_feature_list(self):
        feature_list = {
            "X_train_list": [],
            "X_test_list": [],
        }
        for feature_block in self.feature_blocks:
            X_train = self.X_train[:, feature_block]
            X_test = self.X_test[:, feature_block]
            feature_list["X_train_list"].append(X_train)
            feature_list["X_test_list"].append(X_test)
        return feature_list

    def _get_class_indices(self, class_block):
        train_indices = np.where(np.isin(self.y_train, class_block))
        test_indices = np.where(np.isin(self.y_test, class_block))
        return {
            "class_block": class_block,
            "train_indices": train_indices,
            "test_indices": test_indices,
        }

    def _generate_feature_blocks_assignments(self):
        feature_blocks_assignments = list(
            combinations(range(self.n_feature_splits), self.n_learnware_blocks)
        )
        return feature_blocks_assignments

    def get_sub_dataset(self, feature_block_list, class_block_list):
        # get the feature list
        feature_list = self._get_feature_list()
        X_train_list = feature_list["X_train_list"]
        X_test_list = feature_list["X_test_list"]

        # get the class indices
        if self.task_type == "classification":
            class_list = [self.class_blocks[idx] for idx in class_block_list]
            class_list = [item for sublist in class_list for item in sublist]
            class_indices_info = self._get_class_indices(class_list)
            train_indices = class_indices_info["train_indices"]
            test_indices = class_indices_info["test_indices"]
        else:
            train_indices = np.array(range(len(self.y_train)))
            test_indices = np.array(range(len(self.y_test)))
            class_list = []

        # filter the class
        X_train_list = [X_train[train_indices] for X_train in X_train_list]
        X_test_list = [X_test[test_indices] for X_test in X_test_list]
        X_train_list_filtered = []
        for i in range(len(X_train_list)):
            if i in feature_block_list:
                X_train_list_filtered.append(X_train_list[i])
            else:
                X_train_list_filtered.append(np.array([]))
        X_test_list_filtered = []
        for i in range(len(X_test_list)):
            if i in feature_block_list:
                X_test_list_filtered.append(X_test_list[i])
            else:
                X_test_list_filtered.append(np.array([]))
        y_train = self.y_train[train_indices]
        y_test = self.y_test[test_indices]

        X_train_list_filtered, X_test_list_filtered = self._normalize_X_list(
            X_train_list_filtered, X_test_list_filtered
        )
        if self.task_name == "regression":
            y_train, y_test = self._normalize_y(y_train, y_test)

        return {
            "feature_block_assignment": feature_block_list,
            "class_list": class_list,
            "X_train_list": X_train_list_filtered,
            "X_train": self._merge_X_list(X_train_list_filtered),
            "y_train": y_train,
            "X_test_list": X_test_list_filtered,
            "X_test": self._merge_X_list(X_test_list_filtered),
            "y_test": y_test
        }

    def _merge_X_list(self, X_list):
        non_empty_X_list = [X for X in X_list if X.size > 0]

        # If all arrays are empty, raise an error
        if not non_empty_X_list:
            raise ValueError("All arrays are empty. Cannot merge empty arrays.")

        # Check that all non-empty arrays have the same number of rows
        n_rows = non_empty_X_list[0].shape[0]
        for X in non_empty_X_list:
            if X.shape[0] != n_rows:
                raise ValueError(
                    "All non-empty arrays must have the same number of rows."
                )

        # Concatenate along the second axis (axis=1)
        X_combined = np.concatenate(non_empty_X_list, axis=1)

        return X_combined

    def _normalize_X(self, X_train, X_test):
        if len(X_train) == 0:
            return X_train, X_test
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = fill_nan_with_exception_early(X_train)
        X_test = fill_nan_with_exception_early(X_test)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def _normalize_X_list(self, X_train_list, X_test_list):
        for i in range(len(X_train_list)):
            X_train_list[i], X_test_list[i] = self._normalize_X(
                X_train_list[i], X_test_list[i]
            )
        return X_train_list, X_test_list

    def _normalize_y(self, y_train, y_test):
        scaler = MinMaxScaler()
        scaler.fit(y_train.reshape(-1, 1))
        y_train = scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
        y_test = scaler.transform(y_test.reshape(-1, 1)).reshape(-1)
        return y_train, y_test

    def sample_labeled_data(self, X_train, y_train, n_samples=10, random_state=0):
        if self.task_type == "classification":
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=n_samples, random_state=random_state
            )
            for train_index, _ in sss.split(X_train, y_train):
                selected_idx = train_index
        elif self.task_type == "regression":
            num_buckets = min(10, n_samples)
            y_binned = pd.qcut(y_train, q=num_buckets, labels=False, duplicates='drop')

            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=n_samples, random_state=random_state
            )
            try:
                for train_index, _ in sss.split(X_train, y_binned):
                    selected_idx = train_index
            except ValueError as e:
                self.rng = np.random.default_rng(random_state)
                selected_idx = self.rng.choice(X_train.shape[0], n_samples, replace=False)
                logger.warning(f"Stratified sampling failed, error: {e}. Fallback to random sampling.")
        sample_res = {
            "X_train": X_train[selected_idx],
            "y_train": y_train[selected_idx],
        }
        return sample_res

    def _get_learnware_feature_split_info(self):
        info = {
            "dim_list": [
                len(self.feature_blocks[i]) for i in range(self.n_feature_splits)
            ],
            "feature_block_assignments": [],
        }

        for i in range(len(self.get_learnware_task_ids())):
            feature_block_assignment = self.feature_blocks_assignments[
                i // self.n_class_splits
            ]
            info["feature_block_assignments"].append(feature_block_assignment)
        return info

    def get_learnware_feature_split_info(self):
        return self.learnware_feature_split_info


def split_X(X, dim_list, feature_blocks_assignment, total_feature_blocks):
    # Ensure the sum of dimensions matches the number of columns in X
    if sum(dim_list) != X.shape[1]:
        raise ValueError(
            f"The sum of dimensions in dim_list must equal the number of features in X, currently sum of dim_list: {sum(dim_list)}, number of features in X: {X.shape[1]}"
        )

    # Split the features of X according to the dimensions specified in dim_list
    X_list = []
    start_index = 0
    for dim in dim_list:
        end_index = start_index + dim
        X_list.append(X[:, start_index:end_index])
        start_index = end_index

    # Filter the list to only include blocks specified in feature_blocks_assignment
    X_list_filtered = []
    for i in range(total_feature_blocks):
        if i in feature_blocks_assignment:
            X_list_filtered.append(X_list[feature_blocks_assignment.index(i)])
        else:
            # Append an empty array with the same number of rows as X and zero columns
            X_list_filtered.append(np.array([]))

    return X_list_filtered


def split_feature(X, dim_list, feature_blocks_assignment):
    total_feature_blocks = len(dim_list)
    dim_list_select = [dim_list[i] for i in feature_blocks_assignment]
    total_feature_blocks = len(dim_list)

    num_features = X.shape[1]
    dim_sum_select = sum(dim_list_select)

    # Ensure the sum of dimensions matches the number of columns in X
    if dim_sum_select != num_features:
        raise ValueError(
            f"The sum of dimensions in dim_list must equal the number of features in X, currently sum of dim_list: {dim_sum_select}, number of features in X: {num_features}"
        )

    # Split the features of X according to the dimensions specified in dim_list
    X_list = []
    start_index = 0
    for dim in dim_list_select:
        end_index = start_index + dim
        X_list.append(X[:, start_index:end_index])
        start_index = end_index

    # Filter the list to only include blocks specified in feature_blocks_assignment
    X_list_filtered = []
    for i in range(total_feature_blocks):
        if i in feature_blocks_assignment:
            X_list_filtered.append(X_list[feature_blocks_assignment.index(i)])
        else:
            # Append an empty array with the same number of rows as X and zero columns
            X_list_filtered.append(np.array([]))

    return X_list_filtered