import numpy as np
import pandas as pd

from learnware.specification import (
    RegularStatSpecification,
    generate_rkme_table_spec,
    RKMETableSpecification,
)
from learnware.learnware.base import BaseModel
from learnware.utils import allocate_cuda_idx, choose_device
import torch
import json
import codecs
import os


class LabeledSpecification(RegularStatSpecification):
    def __init__(self, gamma: float = 0.1, cuda_idx: int = None):
        self.gamma = gamma
        self._cuda_idx = cuda_idx if cuda_idx is not None else allocate_cuda_idx()
        self._device = choose_device(cuda_idx=self._cuda_idx)

        self.rkme = None  # type: RKMETableSpecification
        self.y = None  # type: torch.Tensor

        super(LabeledSpecification, self).__init__(type=self.__class__.__name__)

    def generate_stat_spec_from_data(
        self,
        X: np.ndarray,
        model: BaseModel,
        reduced_set_size: int = 50,
        step_size: float = 0.1,
        steps: int = 3,
    ):
        basic_rkme = generate_rkme_table_spec(
            X=X,
            gamma=self.gamma,
            reduced_set_size=reduced_set_size,
            step_size=step_size,
            steps=steps,
            nonnegative_beta=True,
            reduce=True,
            cuda_idx=self._cuda_idx,
        )

        X_reduced = basic_rkme.get_z()
        y_pred = model.predict(X_reduced)
        y_pred = np.ravel(y_pred)

        y_pred = torch.tensor(y_pred, device=self._device).float()

        self.rkme = basic_rkme
        self.y = y_pred

    def get_z(self):
        return self.rkme.get_z()

    def get_beta(self):
        return self.rkme.get_beta()

    def get_y(self):
        return self.y.detach().cpu().numpy()

    def save(self, file_path: str):
        save_path = file_path
        rkme_to_save = self.get_states()
        rkme_states = rkme_to_save["rkme"].get_states()  # Retrieve states of 'rkme'

        # Correct the access to the attributes
        if torch.is_tensor(rkme_states["z"]):
            rkme_states["z"] = (
                rkme_states["z"].detach().cpu().numpy()
            )  # Detach and convert to NumPy array
        rkme_states["z"] = rkme_states[
            "z"
        ].tolist()  # Convert to list for JSON serialization

        if torch.is_tensor(rkme_states["beta"]):
            rkme_states["beta"] = (
                rkme_states["beta"].detach().cpu().numpy()
            )  # Detach and convert to NumPy array
        rkme_states["beta"] = rkme_states[
            "beta"
        ].tolist()  # Convert to list for JSON serialization

        # Update the 'rkme' state in rkme_to_save with modified rkme_states
        rkme_to_save["rkme"] = rkme_states

        # Convert the y array to a list
        rkme_to_save["y"] = rkme_to_save["y"].detach().cpu().numpy().tolist()

        # Write to file
        with codecs.open(save_path, "w", encoding="utf-8") as fout:
            json.dump(rkme_to_save, fout, separators=(",", ":"))

    def load(self, file_path: str):
        # Load JSON file:
        load_path = file_path
        if os.path.exists(load_path):
            with codecs.open(load_path, "r", encoding="utf-8") as fin:
                obj_text = fin.read()
            rkme_load = json.loads(obj_text)
            rkme_load["rkme"]["z"] = torch.from_numpy(
                np.array(rkme_load["rkme"]["z"])
            ).to(self._device)
            rkme_load["rkme"]["beta"] = torch.from_numpy(
                np.array(rkme_load["rkme"]["beta"])
            ).to(self._device)
            rkme_load["y"] = torch.from_numpy(np.array(rkme_load["y"])).to(self._device)

            for d in self.get_states():
                if d in rkme_load.keys():
                    if d == "rkme":
                        rkme = RKMETableSpecification()
                        for k in rkme.get_states():
                            if k in rkme_load[d].keys():
                                if k == "type" and rkme_load[d][k] != rkme.type:
                                    raise TypeError(
                                        f"The type of loaded RKME ({rkme_load[d][k]}) is different from the expected type ({rkme.type})!"
                                    )
                                setattr(rkme, k, rkme_load[d][k])
                        setattr(self, d, rkme)
                    elif d == "type" and rkme_load[d] != self.type:
                        raise TypeError(
                            f"The type of loaded RKME ({rkme_load[d]}) is different from the expected type ({self.type})!"
                        )
                    else:
                        setattr(self, d, rkme_load[d])


class LabeledRKMESpecification(RegularStatSpecification):
    def __init__(self, gamma: float = 0.1, cuda_idx: int = None):
        self.gamma = gamma
        self._cuda_idx = cuda_idx if cuda_idx is not None else allocate_cuda_idx()
        self._device = choose_device(cuda_idx=self._cuda_idx)

        self.rkme = None  # type: RKMETableSpecification
        self.y = None  # type: torch.Tensor

        super(LabeledRKMESpecification, self).__init__(type=self.__class__.__name__)

    def generate_stat_spec_from_data(
        self,
        X: np.ndarray,
        model: BaseModel,
        task_type: str,
        reduced_set_size: int = 50,
        step_size: float = 0.1,
        steps: int = 3,
    ):
        y_pred = model.predict(X)
        X_concat = np.concatenate((X, y_pred.reshape(-1, 1)), axis=1)

        rkme_concat = generate_rkme_table_spec(
            X=X_concat,
            gamma=self.gamma,
            reduced_set_size=reduced_set_size,
            step_size=step_size,
            steps=steps,
            nonnegative_beta=True,
            reduce=True,
            cuda_idx=self._cuda_idx,
        )

        self.rkme = rkme_concat
        self.rkme.z = self.rkme.z[:, :-1]
        self.y = self.rkme.z[:, -1]

        # if task_type == "classification", calculate the number of classes
        if task_type == "classification":
            n_classes = len(np.unique(y_pred))

        # if task_type == "classification", round the y values
        if task_type == "classification":
            self.y = torch.round(self.y)
            # if self.y<0, set it to 0
            self.y = torch.max(self.y, torch.tensor(0, device=self._device).float())
            # if self.y>n_classes-1, set it to n_classes-1
            self.y = torch.min(
                self.y, torch.tensor(n_classes - 1, device=self._device).float()
            )

    def generate_stat_spec_from_labeled_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        reduced_set_size: int = 50,
        step_size: float = 0.1,
        steps: int = 3,
    ):
        X_concat = np.concatenate((X, y.reshape(-1, 1)), axis=1)

        rkme_concat = generate_rkme_table_spec(
            X=X_concat,
            gamma=self.gamma,
            reduced_set_size=reduced_set_size,
            step_size=step_size,
            steps=steps,
            nonnegative_beta=True,
            reduce=True,
            cuda_idx=self._cuda_idx,
        )

        self.rkme = rkme_concat
        self.rkme.z = self.rkme.z[:, :-1]
        self.y = self.rkme.z[:, -1]

        # if task_type == "classification", calculate the number of classes
        if task_type == "classification":
            n_classes = len(np.unique(y))

        # if task_type == "classification", round the y values
        if task_type == "classification":
            self.y = torch.round(self.y)
            # if self.y<0, set it to 0
            self.y = torch.max(self.y, torch.tensor(0, device=self._device).float())
            # if self.y>n_classes-1, set it to n_classes-1
            self.y = torch.min(
                self.y, torch.tensor(n_classes - 1, device=self._device).float()
            )

    def get_z(self):
        return self.rkme.get_z()

    def get_beta(self):
        return self.rkme.get_beta()

    def get_y(self):
        return self.y.detach().cpu().numpy()

    def save(self, file_path: str):
        save_path = file_path
        rkme_to_save = self.get_states()
        rkme_states = rkme_to_save["rkme"].get_states()  # Retrieve states of 'rkme'

        # Correct the access to the attributes
        if torch.is_tensor(rkme_states["z"]):
            rkme_states["z"] = (
                rkme_states["z"].detach().cpu().numpy()
            )  # Detach and convert to NumPy array
        rkme_states["z"] = rkme_states[
            "z"
        ].tolist()  # Convert to list for JSON serialization

        if torch.is_tensor(rkme_states["beta"]):
            rkme_states["beta"] = (
                rkme_states["beta"].detach().cpu().numpy()
            )  # Detach and convert to NumPy array
        rkme_states["beta"] = rkme_states[
            "beta"
        ].tolist()  # Convert to list for JSON serialization

        # Update the 'rkme' state in rkme_to_save with modified rkme_states
        rkme_to_save["rkme"] = rkme_states

        # Convert the y array to a list
        rkme_to_save["y"] = rkme_to_save["y"].detach().cpu().numpy().tolist()

        # Write to file
        with codecs.open(save_path, "w", encoding="utf-8") as fout:
            json.dump(rkme_to_save, fout, separators=(",", ":"))

    def load(self, file_path: str):
        # Load JSON file:
        load_path = file_path
        if os.path.exists(load_path):
            with codecs.open(load_path, "r", encoding="utf-8") as fin:
                obj_text = fin.read()
            rkme_load = json.loads(obj_text)
            rkme_load["rkme"]["z"] = torch.from_numpy(
                np.array(rkme_load["rkme"]["z"])
            ).to(self._device)
            rkme_load["rkme"]["beta"] = torch.from_numpy(
                np.array(rkme_load["rkme"]["beta"])
            ).to(self._device)
            rkme_load["y"] = torch.from_numpy(np.array(rkme_load["y"])).to(self._device)

            for d in self.get_states():
                if d in rkme_load.keys():
                    if d == "rkme":
                        rkme = RKMETableSpecification()
                        for k in rkme.get_states():
                            if k in rkme_load[d].keys():
                                if k == "type" and rkme_load[d][k] != rkme.type:
                                    raise TypeError(
                                        f"The type of loaded RKME ({rkme_load[d][k]}) is different from the expected type ({rkme.type})!"
                                    )
                                setattr(rkme, k, rkme_load[d][k])
                        setattr(self, d, rkme)
                    elif d == "type" and rkme_load[d] != self.type:
                        raise TypeError(
                            f"The type of loaded RKME ({rkme_load[d]}) is different from the expected type ({self.type})!"
                        )
                    else:
                        setattr(self, d, rkme_load[d])


class SystemSpecification(RegularStatSpecification):

    def __init__(self, cuda_idx: int = None, device: torch.device = None):
        self.x = None
        self.y = None
        self.weights = None
        self.gamma = 0.1

        if device is None:
            self._cuda_idx = cuda_idx if cuda_idx is not None else allocate_cuda_idx()
            self._device = choose_device(cuda_idx=self._cuda_idx)
        else:
            self._device = device

        super(SystemSpecification, self).__init__(type=self.__class__.__name__)

    def generate_stat_spec_from_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        weights: torch.Tensor,
        gamma: float = 0.1,
    ):
        self.x = X.to(self._device)
        self.y = y.to(self._device)
        self.weights = weights.to(self._device)
        self.gamma = gamma

    def get_z(self):
        return self.x.detach().cpu().numpy()

    def get_y(self):
        return self.y.detach().cpu().numpy()

    def get_beta(self):
        return self.weights.detach().cpu().numpy()

    def generate_stat_spec_from_spec(self, labeled_spec: LabeledSpecification):
        self.x = labeled_spec.rkme.z.to(self._device)
        self.y = labeled_spec.y.to(self._device)
        self.weights = labeled_spec.rkme.beta.to(self._device)
        self.gamma = labeled_spec.gamma

    def save(self, file_path: str):
        spec_to_save = self._get_states()
        spec_to_save["x"] = spec_to_save["x"].detach().cpu().numpy().tolist()
        spec_to_save["y"] = spec_to_save["y"].detach().cpu().numpy().tolist()
        spec_to_save["weights"] = (
            spec_to_save["weights"].detach().cpu().numpy().tolist()
        )
        with codecs.open(file_path, "w", encoding="utf-8") as fout:
            json.dump(spec_to_save, fout, separators=(",", ":"))

    def load(self, file_path: str):
        if os.path.exists(file_path):
            with codecs.open(file_path, "r", encoding="utf-8") as fin:
                obj_text = fin.read()
            spec_load = json.loads(obj_text)
            spec_load["x"] = torch.from_numpy(np.array(spec_load["x"])).to(self._device)
            spec_load["y"] = torch.from_numpy(np.array(spec_load["y"])).to(self._device)
            spec_load["weights"] = torch.from_numpy(np.array(spec_load["weights"])).to(
                self._device
            )

            for d in self.get_states():
                if d in spec_load.keys():
                    setattr(self, d, spec_load[d])

    def _get_states(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}