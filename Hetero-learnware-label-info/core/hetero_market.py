from learnware.learnware import Learnware
from .specification import (
    LabeledSpecification,
    LabeledRKMESpecification,
    SystemSpecification,
    MMDDistance,
    LabeledSpecificationCLS,
)
from learnware.specification import RKMETableSpecification
from typing import List
from .subspace import (
    train_epoch_multi,
    AutoEncoder,
    AEDataset,
    TableResNet
)
from ..benchmarks import split_feature
from ..utils import fix_seed

from torch.utils.data import DataLoader
from learnware.utils import allocate_cuda_idx, choose_device
from torch.optim import Adam
from loguru import logger
import os
import torch
import numpy as np
import json
import hashlib
from typing import Union


class HeteroLearnwareMarket:
    def __init__(
        self,
        learnware_list: List[Learnware],
        task_type: str,
        learnware_feature_split_info: dict,
        training_args: dict = None,
        matching_args: dict = None,
        learnware_args: dict = None,
        cuda_idx: int = None,
        ckpt_dir: str = None,
        seed: int = 0,
    ) -> None:
        fix_seed(seed)
        self.learnware_list = learnware_list
        self.task_type = task_type
        self.cuda_idx = allocate_cuda_idx() if cuda_idx is None else cuda_idx
        self.device = choose_device(cuda_idx=self.cuda_idx)

        self.learnware_feature_split_info = learnware_feature_split_info

        default_training_args = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "epochs": 1000,
            "architecture": {
                "hidden_dim": 32,
                "subspace_dim": 16,
            },
            "loss_weights": {
                "weight_cont": 1.0,
                "weight_cont_supervised": 1.0,
                "weight_rec": 1.0,
                "weight_supervised": 1.0,
                "weight_global": 0.1,
                "use_weight": True,
            },
            "mmd_calculator_paras": {
                "task_type": task_type,
                "gamma": 0.1,
                "cuda_idx": self.cuda_idx,
                "cond_mmd_coef": 1,
                "num_bins": 10,
            },
            "rkme_for_training": "LabeledSpecificationCLS",
        }
        if training_args is not None:
            self.training_args = self._merge_dicts(default_training_args, training_args)
        self.training_args = default_training_args

        default_matching_args = {
            "gamma": 0.1,
            "num_bins": 10,
            "cond_mmd_coef": 1,
        }
        if matching_args is not None:
            self.matching_args = self._merge_dicts(default_matching_args, matching_args)
        self.matching_args = default_matching_args

        self.subspace_dim = self.training_args["architecture"]["subspace_dim"]

        Setup = {
            "Learnware args": learnware_args,
            "Training args": self.training_args,
            "Matching args": self.matching_args,
        }
        setup_str = json.dumps(Setup, sort_keys=True)
        setup_hash = hashlib.md5(setup_str.encode("utf-8")).hexdigest()
        self.current_ckpt_dir = os.path.join(ckpt_dir, f"ckpt_{setup_hash}")
        os.makedirs(self.current_ckpt_dir, exist_ok=True)

    def _save_all_models(self):
        # save autoencdoers, system engine
        for i, auto_encoder in enumerate(self.auto_encoders):
            auto_encoder.eval()
            model_path = os.path.join(self.current_ckpt_dir, f"autoencoder_{i}.pt")
            torch.save(auto_encoder.state_dict(), model_path)
        model_path = os.path.join(self.current_ckpt_dir, "system_engine.pt")
        self.system_engine.eval()
        torch.save(self.system_engine.state_dict(), model_path)

    def _get_all_ckpt_paths(self):
        autoencoder_paths = [
            os.path.join(self.current_ckpt_dir, f"autoencoder_{i}.pt")
            for i in range(len(self.auto_encoders))
        ]
        system_engine_path = os.path.join(self.current_ckpt_dir, "system_engine.pt")
        return autoencoder_paths, system_engine_path

    def _check_all_ckpts(self):
        autoencoder_paths, system_engine_path = self._get_all_ckpt_paths()
        if all([os.path.exists(path) for path in autoencoder_paths]) and os.path.exists(
            system_engine_path
        ):
            return True
        else:
            return False

    def _load_all_models(self):
        # check the all paths, if all exists, load autoencoders and system engine
        autoencoder_paths, system_engine_path = self._get_all_ckpt_paths()
        for i, auto_encoder in enumerate(self.auto_encoders):
            auto_encoder.load_state_dict(torch.load(autoencoder_paths[i]))
        self.system_engine.load_state_dict(torch.load(system_engine_path))

    @classmethod
    def _merge_dicts(cls, defaults, updates):
        """
        Merges two dictionaries where nested dictionaries are updated key by key,
        instead of being entirely replaced. This function is particularly useful
        for updating nested configurations such as training parameters and loss weights.

        Parameters
        ----------
        defaults : dict
            The default dictionary containing initial settings.
        updates : dict
            The updates dictionary containing new values, which may overlap with the defaults.

        Returns
        -------
        dict
            A merged dictionary with updates applied over the defaults.

        """
        if updates:
            for key, value in updates.items():
                if isinstance(value, dict) and key in defaults:
                    # Recursively merge dictionaries for nested structures
                    defaults[key] = cls._merge_dicts(defaults.get(key, {}), value)
                else:
                    # Update values directly for non-dictionary types
                    defaults[key] = value
        return defaults

    def _get_dataloaders(self):
        dataloaders = []
        for learnware in self.learnware_list:
            if self.training_args["rkme_for_training"] == "LabeledSpecification":
                labeled_spec = learnware.get_specification().get_stat_spec_by_name(
                    "LabeledSpecification"
                )
            elif self.training_args["rkme_for_training"] == "LabeledRKMESpecification":
                labeled_spec = learnware.get_specification().get_stat_spec_by_name(
                    "LabeledRKMESpecification"
                )
            elif self.training_args["rkme_for_training"] == "LabeledSpecificationCLS":
                labeled_spec = learnware.get_specification().get_stat_spec_by_name(
                    "LabeledSpecificationCLS"
                )
            elif self.training_args["rkme_for_training"] == "RKMETableSpecification":
                labeled_spec = learnware.get_specification().get_stat_spec_by_name(
                    "RKMETableSpecification"
                )
            z = labeled_spec.get_z()
            weights = labeled_spec.get_beta()
            if self.training_args["rkme_for_training"] != "RKMETableSpecification":
                y = labeled_spec.get_y()
            else:
                y = np.zeros(z.shape[0])
            train_ds = AEDataset(
                data=z, target=y, weights=weights, task_type=self.task_type
            )
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
            dataloaders.append(train_loader)
        return dataloaders

    def _get_autoencoders(self):
        dim_list = self.learnware_feature_split_info["dim_list"]
        autoencoders = []
        for i in range(len(dim_list)):
            autoencoder = AutoEncoder(
                input_dim=dim_list[i],
                hidden_dim=self.training_args["architecture"]["hidden_dim"],
                encoding_dim=self.subspace_dim,
            ).to(self.device)
            autoencoders.append(autoencoder)
        return autoencoders

    def _get_task_classes(self):
        if self.task_type == "classification":
            all_labels = []
            for learnware in self.learnware_list:
                labeled_spec = learnware.get_specification().get_stat_spec_by_name(
                    "LabeledSpecificationCLS"
                )
                y = labeled_spec.get_y()
                all_labels.extend(y)

            n_classes = len(set(all_labels))
            return n_classes
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for function: _get_task_classes"
            )

    def _get_engine_model_output_dim(self):
        if self.task_type == "classification":
            return self._get_task_classes()
        elif self.task_type == "regression":
            return 1
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for function: _get_engine_model_output_dim"
            )

    def _generate_engine_model_architecture(self):
        engine_model = TableResNet(
            input_dim=self.subspace_dim,
            hidden_dim=self.training_args["architecture"]["hidden_dim"],
            output_dim=self._get_engine_model_output_dim(),
        ).to(self.device)
        return engine_model

    def _get_all_training_elements(self):
        training_elements_all = {
            "train_loaders": self._get_dataloaders(),
            "autoencoders": self._get_autoencoders(),
        }
        return training_elements_all

    def subspace_learning(self, regenerate=False):
        training_elements_all = self._get_all_training_elements()
        models = training_elements_all["autoencoders"]
        train_loaders = training_elements_all["train_loaders"]
        engine_model = self._generate_engine_model_architecture()
        self.auto_encoders = models
        self.system_engine = engine_model

        if self._check_all_ckpts() and not regenerate:
            self._load_all_models()
            logger.info(
                f"Loaded all models from checkpoint, ckpt dir:{self.current_ckpt_dir}"
            )
        else:
            logger.info("Start training.")

            paras_list = []
            for model in models:
                paras_list.extend(list(model.parameters()))
            paras_list.extend(list(engine_model.parameters()))
            optimizer = Adam(
                paras_list,
                lr=self.training_args["lr"],
                weight_decay=self.training_args["weight_decay"],
            )

            loss_history = []

            epochs = self.training_args["epochs"]
            for epoch in range(1, epochs + 1):
                epoch_loss = train_epoch_multi(
                    models=models,
                    system_engine=engine_model,
                    train_loaders=train_loaders,
                    optimizer=optimizer,
                    device=self.device,
                    task_type=self.task_type,
                    learnware_feature_split_info=self.learnware_feature_split_info,
                    loss_weights=self.training_args["loss_weights"],
                    mmd_calculator_paras=self.training_args["mmd_calculator_paras"],
                )
                loss_history.append(epoch_loss)

                if epoch % 10 == 0:
                    logger.info(
                        f"epoch {epoch}/{epochs} - loss: {loss_history[-1]:.4f}"
                    )

            self.auto_encoders = models
            self.system_engine = engine_model

            self._save_all_models()

        # set the autoencoders and system engine to eval mode
        for model in self.auto_encoders:
            model.eval()
        self.system_engine.eval()

        self._generate_system_specifications()

    def _generate_system_specification(
        self,
        raw_specification: Union[
            LabeledSpecificationCLS,
            LabeledSpecification,
            RKMETableSpecification,
            LabeledRKMESpecification,
        ],
        feature_blocks_assignment: list,
    ) -> SystemSpecification:
        if raw_specification.type == "LabeledSpecificationCLS":
            x = raw_specification.z
            x_torch = torch.tensor(x).to(self.device).float()
            x_embedded = self._get_embedding(x_torch, feature_blocks_assignment)
            y = raw_specification.y
            weights = raw_specification.beta
        elif raw_specification.type == "LabeledSpecification":
            x = raw_specification.rkme.z
            x_torch = torch.tensor(x).to(self.device).float()
            x_embedded = self._get_embedding(x_torch, feature_blocks_assignment)
            y = raw_specification.y
            weights = raw_specification.rkme.beta
        elif raw_specification.type == "LabeledRKMESpecification":
            x = raw_specification.rkme.z
            x_torch = torch.tensor(x).to(self.device).float()
            x_embedded = self._get_embedding(x_torch, feature_blocks_assignment)
            y = raw_specification.y
            weights = raw_specification.rkme.beta
        elif raw_specification.type == "RKMETableSpecification":
            x = raw_specification.z
            x_torch = torch.tensor(x).to(self.device).float()
            x_embedded = self._get_embedding(x_torch, feature_blocks_assignment)
            y = torch.zeros(x.shape[0]).to(self.device)
            weights = raw_specification.beta
        system_spec = SystemSpecification(cuda_idx=self.cuda_idx)
        system_spec.generate_stat_spec_from_data(x_embedded, y, weights)
        return system_spec

    def _generate_system_requirement(
        self,
        raw_requirement: Union[
            LabeledSpecificationCLS,
            LabeledSpecification,
            RKMETableSpecification,
            LabeledRKMESpecification,
        ],
        feature_blocks_assignment: list,
    ) -> SystemSpecification:
        if raw_requirement.type == "LabeledSpecificationCLS":
            x = raw_requirement.z
            x_torch = torch.tensor(x).to(self.device).float()
            x_embedded = self._get_embedding(x_torch, feature_blocks_assignment)
            y = raw_requirement.y
            weights = raw_requirement.beta
        elif raw_requirement.type == "LabeledSpecification":
            x = raw_requirement.rkme.z
            x_torch = torch.tensor(x).to(self.device).float()
            x_embedded = self._get_embedding(x_torch, feature_blocks_assignment)
            y = raw_requirement.y
            weights = raw_requirement.rkme.beta
        elif raw_requirement.type == "LabeledRKMESpecification":
            x = raw_requirement.rkme.z
            x_torch = torch.tensor(x).to(self.device).float()
            x_embedded = self._get_embedding(x_torch, feature_blocks_assignment)
            y = raw_requirement.y
            weights = raw_requirement.rkme.beta
        elif raw_requirement.type == "RKMETableSpecification":
            x = raw_requirement.z
            x_torch = torch.tensor(x).to(self.device).float()
            x_embedded = self._get_embedding(x_torch, feature_blocks_assignment)
            y = torch.zeros(x.shape[0]).to(self.device)
            weights = raw_requirement.beta
        system_req = SystemSpecification(cuda_idx=self.cuda_idx)
        system_req.generate_stat_spec_from_data(x_embedded, y, weights)
        return system_req

    def _get_embedding(self, x: torch.Tensor, feature_blocks_assignment: list):
        x_list = split_feature(
            X=x,
            dim_list=self.learnware_feature_split_info["dim_list"],
            feature_blocks_assignment=feature_blocks_assignment,
        )
        x_emb_list = []
        for i in feature_blocks_assignment:
            x_temp = x_list[i]
            auto_encoder = self.auto_encoders[i]
            x_temp_emb = auto_encoder.encode_features(x_temp)
            x_emb_list.append(x_temp_emb)
        # calculate the mean of x_emb_list
        x_emb_tensor = torch.stack(
            x_emb_list
        )  # Stack the list of tensors to create a new dimension
        mean_emb = torch.mean(
            x_emb_tensor, dim=0
        )  # Compute the mean along the new dimension
        return mean_emb

    def _reconstruct_embedding(self, x: torch.Tensor, feature_block_id: int):
        return self.auto_encoders[feature_block_id].decode_features(x)

    def _generate_system_specifications(self):
        for i, learnware in enumerate(self.learnware_list):
            spec = learnware.get_specification()
            if self.training_args["rkme_for_training"] == "LabeledSpecification":
                labeled_spec = spec.get_stat_spec_by_name("LabeledSpecification")
            elif self.training_args["rkme_for_training"] == "LabeledRKMESpecification":
                labeled_spec = spec.get_stat_spec_by_name("LabeledRKMESpecification")
            elif self.training_args["rkme_for_training"] == "LabeledSpecificationCLS":
                labeled_spec = spec.get_stat_spec_by_name("LabeledSpecificationCLS")
            elif self.training_args["rkme_for_training"] == "RKMETableSpecification":
                labeled_spec = spec.get_stat_spec_by_name("RKMETableSpecification")
            system_spec = self._generate_system_specification(
                labeled_spec,
                self.learnware_feature_split_info["feature_block_assignments"][i],
            )
            learnware.update_stat_spec(system_spec.type, system_spec)

    def _recommend_single_learnware(
        self, projected_user_requirement: SystemSpecification
    ) -> Learnware:
        mmd_calculator = MMDDistance(
            task_type=self.task_type,
            gamma=self.matching_args["gamma"],
            cuda_idx=self.cuda_idx,
            cond_mmd_coef=self.matching_args["cond_mmd_coef"],
            num_bins=self.matching_args["num_bins"],
        )

        learnware_rkmes = []
        for learnware in self.learnware_list:
            system_spec = learnware.get_specification().get_stat_spec_by_name(
                "SystemSpecification"
            )
            learnware_rkmes.append(system_spec)

        mmds = []
        for learnware_rkme in learnware_rkmes:
            mmd = mmd_calculator.mmd_total(projected_user_requirement, learnware_rkme)
            mmds.append(mmd)

        # select the learnware with minumum mmd
        best_learnware_idx = mmds.index(min(mmds))
        return self.learnware_list[best_learnware_idx], best_learnware_idx

    def recommend(
        self,
        user_requirement: Union[
            LabeledSpecificationCLS,
            LabeledSpecification,
            RKMETableSpecification,
            LabeledRKMESpecification,
        ],
        feature_blocks_assignment: list,
    ) -> Learnware:
        system_requirement = self._generate_system_requirement(
            user_requirement, feature_blocks_assignment
        )
        best_learnware, best_learnware_idx = self._recommend_single_learnware(
            system_requirement
        )

        return best_learnware, best_learnware_idx