import numpy as np
import os

from .tabzilla import TabzillaLoader
from .split_dataset import TaskGenerator
from .config import LEARNWARE_FOLDER_DIR, LEARNWARE_ZIP_DIR, MODEL_DIR
from .templates import JoblibModelTemplate, OurLearnwareTemplate
from .benchmark_utils import get_learnware_from_dirpath, evaluate_loss
from ..utils import train_lgb_model, fix_seed


from tqdm import tqdm
import tempfile
from learnware.specification import generate_stat_spec
from learnware.tests.templates import StatSpecTemplate
import zipfile
import joblib
from loguru import logger

import shutil


class LearnwareBenchmark:
    def __init__(self, dataset_name, n_splits=4, n_learnware_blocks=3, seed=0) -> None:
        self.dataset_name = dataset_name

        self.data_loader = TabzillaLoader()

        split_dataset = self.data_loader.get_split_dataset(dataset_name)
        self.task_type = split_dataset["task_type"]
        self.task_generator = TaskGenerator(
            split_data=split_dataset,
            n_feature_splits=n_splits,
            n_learnware_blocks=n_learnware_blocks,
            seed=seed,
        )
        self.seed = seed

        self.learnware_task_ids = self.task_generator.get_learnware_task_ids()
        self.user_task_ids = self.task_generator.get_all_user_task_ids()  # dict
        logger.info(f"learnware_task_ids: {self.learnware_task_ids}")

        self.model_dir = os.path.join(MODEL_DIR, self.dataset_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # check models
        if not self._check_all_learnware_models():
            logger.info(f"Regenerating all learnware models for {dataset_name}..")
            self.regenerate_all_learnware_models()

        # check learnwares
        if not self.check_all_learnwares():
            logger.info(f"Regenerating all learnwares for {dataset_name}..")
            self.regenerate_all_learnwares()
    
    def get_task_type(self):
        return self.task_type

    def get_learnware_task_ids(self):
        return self.learnware_task_ids

    def get_user_task_ids(self):
        return self.user_task_ids

    def get_learnware_task_dataset_by_id(self, learnware_task_id):
        return self.task_generator.get_learnware_task_by_id(learnware_task_id)

    def get_user_task_dataset_by_id(self, user_task_id, n_mix):
        return self.task_generator.get_user_task_by_id(user_task_id, n_mix)

    def sample_labeled_data(self, X_train, y_train, n_samples=10, random_state=0):
        return self.task_generator.sample_labeled_data(
            X_train, y_train, n_samples, random_state=random_state
        )

    def get_learnware_feature_split_info(self):
        return self.task_generator.get_learnware_feature_split_info()

    def get_model_by_id(self, learnware_id):
        model_path = os.path.join(self.model_dir, f"{learnware_id}.out")
        if not os.path.exists(model_path):
            self._regenerate_learnware_model(learnware_id)
        model = joblib.load(model_path)
        return model

    def get_learnware_by_id(
        self, learnware_id, reduced_set_size=50, cuda_idx=0, regenerate_flag=False
    ):
        from ..core.specification import (
            LabeledSpecification,
            LabeledSpecificationCLS,
            LabeledRKMESpecification,
        )

        if learnware_id not in self.learnware_task_ids:
            raise ValueError(
                f"learnware_id={learnware_id} is not in learnware_task_ids"
            )

        learnware_folderpath = os.path.join(
            LEARNWARE_FOLDER_DIR, self.dataset_name, f"{learnware_id}"
        )

        if regenerate_flag or not os.path.exists(learnware_folderpath):
            learnware_zippath = os.path.join(
                LEARNWARE_ZIP_DIR, self.dataset_name, f"{learnware_id}.zip"
            )
            os.makedirs(os.path.dirname(learnware_zippath), exist_ok=True)
            model_path = os.path.join(self.model_dir, f"{learnware_id}.out")

            learnware_data = self.get_learnware_task_dataset_by_id(learnware_id)
            X_train, y_train = learnware_data["X_train"], learnware_data["y_train"]
            input_shape = X_train.shape[1]
            output_shape = len(np.unique(y_train))

            model = joblib.load(model_path)
            y_prediction = model.predict(X_train)

            with tempfile.TemporaryDirectory(
                suffix=f"{self.dataset_name}_{learnware_id}_spec"
            ) as tempdir:
                fix_seed(self.seed)
                basic_spec_file_path = os.path.join(tempdir, "basic_rkme.json")
                basic_rkme = generate_stat_spec(
                    type="table", X=X_train, reduced_set_size=reduced_set_size
                )
                basic_rkme.save(basic_spec_file_path)

                if self.task_type == "classification":
                    fix_seed(self.seed)
                    labeled_rkme_file_path = os.path.join(tempdir, "labeled_rkme.json")
                    labeled_rkme = LabeledSpecificationCLS(cuda_idx=cuda_idx)
                    labeled_rkme.generate_stat_spec_from_data(
                        X=X_train, y=y_prediction, K=reduced_set_size
                    )
                    labeled_rkme.save(labeled_rkme_file_path)

                fix_seed(self.seed)
                model_rkme_file_path = os.path.join(tempdir, "model_rkme.json")
                model_rkme = LabeledSpecification(cuda_idx=cuda_idx)
                model_rkme.generate_stat_spec_from_data(
                    X=X_train, model=model, reduced_set_size=reduced_set_size
                )
                model_rkme.save(model_rkme_file_path)

                fix_seed(self.seed)
                basic_labeled_rkme_file_path = os.path.join(
                    tempdir, "basic_labeled_rkme.json"
                )
                basic_labeled_rkme = LabeledRKMESpecification(cuda_idx=cuda_idx)
                basic_labeled_rkme.generate_stat_spec_from_data(
                    X=X_train,
                    model=model,
                    reduced_set_size=reduced_set_size,
                    task_type=self.task_type,
                )
                basic_labeled_rkme.save(basic_labeled_rkme_file_path)

                if self.task_type == "classification":
                    stat_spec_templates = [
                        StatSpecTemplate(
                            filepath=basic_spec_file_path, type=basic_rkme.type
                        ),
                        StatSpecTemplate(
                            filepath=labeled_rkme_file_path, type=labeled_rkme.type
                        ),
                        StatSpecTemplate(
                            filepath=model_rkme_file_path, type=model_rkme.type
                        ),
                        StatSpecTemplate(
                            filepath=basic_labeled_rkme_file_path,
                            type=basic_labeled_rkme.type,
                        ),
                    ]
                else:
                    stat_spec_templates = [
                        StatSpecTemplate(
                            filepath=basic_spec_file_path, type=basic_rkme.type
                        ),
                        StatSpecTemplate(
                            filepath=model_rkme_file_path, type=model_rkme.type
                        ),
                        StatSpecTemplate(
                            filepath=basic_labeled_rkme_file_path,
                            type=basic_labeled_rkme.type,
                        ),
                    ]
                OurLearnwareTemplate.generate_learnware_zipfile(
                    learnware_zippath=learnware_zippath,
                    model_template=JoblibModelTemplate(
                        model_filepath=model_path,
                        model_kwargs={
                            "input_shape": (input_shape,),
                            "output_shape": (output_shape,),
                            "predict_method": (
                                "predict_proba"
                                if self.task_type == "classification"
                                else "predict"
                            ),
                            "model_filename": os.path.basename(model_path),
                        },
                    ),
                    stat_spec_templates=stat_spec_templates,
                    requirements=["lightgbm==4.3.0"],
                )

            os.makedirs(learnware_folderpath, exist_ok=True)
            with zipfile.ZipFile(learnware_zippath, "r") as zip_ref:
                zip_ref.extractall(learnware_folderpath)

        return get_learnware_from_dirpath(learnware_folderpath)

    def _regenerate_learnware_model(self, idx):
        if idx not in self.learnware_task_ids:
            raise ValueError(f"idx={idx} is not in learnware_task_ids")
        task = self.get_learnware_task_dataset_by_id(idx)
        X_train, y_train, X_test, y_test = (
            task["X_train"],
            task["y_train"],
            task["X_test"],
            task["y_test"],
        )
        print(
            f"train the model for task {idx} with X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}"
        )
        model_path = os.path.join(self.model_dir, f"{idx}.out")
        model = train_lgb_model(X_train, y_train, self.task_type, model_path)

        if self.task_type == "classification":
            pred_y = model.predict_proba(X_test)
        else:
            pred_y = model.predict(X_test)

        loss = evaluate_loss(self.task_type, y_test, pred_y, None)

        print(f"Task {idx} performance: {loss}\n")

    def _check_all_learnware_models(self):
        for idx in self.learnware_task_ids:
            model_path = os.path.join(self.model_dir, f"{idx}.out")
            if not os.path.exists(model_path):
                return False
        return True

    def evaluate(self, y_true, y_pred, model_classes):
        return evaluate_loss(self.task_type, y_true, y_pred, model_classes)

    @staticmethod
    def clear_folder(folder_path):
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"The folder {folder_path} does not exist.")

    def regenerate_all_learnware_models(self):
        self.clear_folder(self.model_dir)
        logger.info(
            f"Clear the model folder: {self.model_dir}, regenerate all models.."
        )
        idx_list = self.get_learnware_task_ids()
        for idx in tqdm(
            idx_list,
            desc="Processing tasks",
            unit="task",
            leave=True,
            bar_format="{l_bar}{bar}{r_bar}",
        ):
            self._regenerate_learnware_model(idx)

    def regenerate_all_learnwares(self, reduced_set_size=50):
        learnware_folderpath_root = os.path.join(
            LEARNWARE_FOLDER_DIR, self.dataset_name
        )
        self.clear_folder(learnware_folderpath_root)
        logger.info(
            f"Clear the learnware folder: {learnware_folderpath_root}, regenerate all learnwares.."
        )

        learnware_task_ids = self.get_learnware_task_ids()
        for task_id in learnware_task_ids:
            self.get_learnware_by_id(
                task_id, reduced_set_size=reduced_set_size, regenerate_flag=True
            )

    def check_all_learnwares(self):
        # check whether all learnwares are available
        learnware_task_ids = self.get_learnware_task_ids()
        for task_id in learnware_task_ids:
            learnware_folderpath = os.path.join(
                LEARNWARE_FOLDER_DIR, self.dataset_name, f"{task_id}"
            )
            if not os.path.exists(learnware_folderpath):
                return False
        return True