import os
import copy

from learnware.specification import RKMETableSpecification

from .benchmarks import *
from .core import *
from .utils import *
from .model_configs import *
from .config import *

class Workflow:
    def __init__(self, task_type: str, regenerate_flag=False):
        self.task_type = task_type
        self.regenerate_flag = regenerate_flag
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.res_dir = os.path.join(self.root_dir, "res")
        self.ckpt_dir = os.path.join(self.res_dir, "ckpt")
        self.lgb_dir = os.path.join(self.res_dir, "lightgbm")

        experiment_setup = ExperimentSetup()
        self.setup = experiment_setup.get_setup_paras(self.task_type)

        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.lgb_dir, exist_ok=True)
    
    def load_market(self, task_name: str):
        self.task_name = task_name
        learnware_args = self.setup["Learnware arguments"]
        training_args = self.setup["Training arguments"]
        matching_args = self.setup["Matching arguments"]

        benchmark = LearnwareBenchmark(
            task_name,
            n_splits=learnware_args["n_splits"],
            n_learnware_blocks=learnware_args["n_learnware_blocks"],
        )

        if self.regenerate_flag:
            benchmark.regenerate_all_learnware_models()
            benchmark.regenerate_all_learnwares(
                reduced_set_size=learnware_args["reduced_set_size"]
            )
        
        # generate the heterogeneous learnware market
        learnware_list = []
        learnware_task_ids = benchmark.get_learnware_task_ids()
        for learnware_task_id in learnware_task_ids:
            learnware = benchmark.get_learnware_by_id(learnware_task_id)
            learnware_list.append(learnware)
        
        learnware_feature_split_info = benchmark.get_learnware_feature_split_info()

        hetero_market = HeteroLearnwareMarket(
            learnware_list=learnware_list,
            task_type=benchmark.get_task_type(),
            learnware_feature_split_info=learnware_feature_split_info,
            training_args=training_args,
            matching_args=matching_args,
            learnware_args=learnware_args,
            ckpt_dir=os.path.join(self.ckpt_dir, task_name)
        )

        hetero_market.subspace_learning(self.regenerate_flag)

        self.benchmark = benchmark
        self.hetero_market = hetero_market
    
    def run_our_method(self):
        performance_learnware_list = []
        performance_user_list = []
        performance_ensemble_user_list = []

        learnware_feature_split_info = self.benchmark.get_learnware_feature_split_info()
        user_task_ids = self.benchmark.get_user_task_ids()
        user_task_paras = self.setup["User task parameters"]
        training_args = self.setup["Training arguments"]
        
        self_training_model_dir = os.path.join(self.lgb_dir, self.task_name)
        os.makedirs(self_training_model_dir, exist_ok=True)

        for n_mix in user_task_paras["n_mix_list"]:
            for user_task_id in user_task_ids[n_mix]:
                user_task, feature_block_list = self.benchmark.get_user_task_dataset_by_id(
                    user_task_id, n_mix=n_mix
                )
                X_train, y_train, X_test, y_test = (
                    user_task["X_train"],
                    user_task["y_train"],
                    user_task["X_test"],
                    user_task["y_test"],
                )

                fix_seed(0)
                temp_performance_learnware_list = []
                temp_performance_user_list = []
                temp_performance_ensemble_user_list = []

                for repeat_idx in range(user_task_paras["n_repeated"]):
                    sample_size = user_task_paras["n_sample_labeled_data"]
                    sample_res = self.benchmark.sample_labeled_data(
                        X_train, y_train, n_samples=sample_size, random_state=repeat_idx
                    )
                    X_train_temp, y_train_temp = (
                        sample_res["X_train"],
                        sample_res["y_train"],
                    )

                    user_lgm_model = train_lgb_model(
                        X_train_temp,
                        y_train_temp,
                        task_type=self.benchmark.get_task_type(),
                        save_path=os.path.join(
                            self_training_model_dir,
                            f"{user_task_id}_{sample_size}_{repeat_idx}.pkl",
                        ),
                        regenerate=self.regenerate_flag,
                    )

                    if self.benchmark.get_task_type() == "classification":
                        user_prediction = user_lgm_model.predict_proba(X_test)
                        user_prediction = complete_y_user(
                            user_prediction,
                            user_lgm_model.classes_,
                            list(range(len(set(y_test)))),
                        )
                    else:
                        user_prediction = user_lgm_model.predict(X_test)

                    # user generate the requirement
                    fix_seed(0)
                    if training_args["rkme_for_training"] == "LabeledSpecificationCLS":
                        user_requirement = LabeledSpecificationCLS()
                        user_requirement.generate_stat_spec_from_ssl_data(
                            X_unlabel=X_test,
                            X_label=X_train_temp,
                            y_label=y_train_temp,
                            reduce=user_task_paras["reduce"],
                            K=user_task_paras["user_reduced_set_size"],
                        )
                    elif training_args["rkme_for_training"] == "LabeledSpecification":
                        user_requirement = LabeledSpecification()
                        user_requirement.generate_stat_spec_from_data(
                            X=X_test,
                            model=user_lgm_model,
                            reduced_set_size=user_task_paras["user_reduced_set_size"],
                        )
                    elif training_args["rkme_for_training"] == "RKMETableSpecification":
                        user_requirement = RKMETableSpecification()
                        user_requirement.generate_stat_spec_from_data(
                            X=X_test,
                            K=user_task_paras["user_reduced_set_size"],
                        )

                    # market recommend the learnware
                    recommend_learnware, best_learnware_idx = self.hetero_market.recommend(
                        user_requirement=user_requirement,
                        feature_blocks_assignment=feature_block_list,
                    )

                    # user reuse the learnware
                    results = learnware_reuse(
                        learnware=recommend_learnware,
                        learnware_block_assignment=learnware_feature_split_info[
                            "feature_block_assignments"
                        ][best_learnware_idx],
                        learnware_market=self.hetero_market,
                        x_user=X_test,
                        user_feature_block_assignment=feature_block_list,
                        dim_list=learnware_feature_split_info["dim_list"],
                        task_type=self.benchmark.get_task_type(),
                        device=self.hetero_market.device,
                        y_user=user_prediction,
                    )

                    y_pre_learnware = results["y_learnware"]
                    y_pre_ensemble_user = results["y_ensemble_with_user"]
                    performance_learnware = self.benchmark.evaluate(
                        y_test, y_pre_learnware, None
                    )
                    performance_user = self.benchmark.evaluate(y_test, user_prediction, None)
                    performance_ensemble_user = self.benchmark.evaluate(
                        y_test, y_pre_ensemble_user, None
                    )

                    temp_performance_learnware_list.append(performance_learnware)
                    temp_performance_user_list.append(performance_user)
                    temp_performance_ensemble_user_list.append(performance_ensemble_user)

                performance_learnware_list.append(temp_performance_learnware_list)
                performance_user_list.append(temp_performance_user_list)
                performance_ensemble_user_list.append(temp_performance_ensemble_user_list)

        results = {
            "Our method": calculate_mean_results(performance_learnware_list),
            "Self-training": calculate_mean_results(performance_user_list),
            "Ensemble with user": calculate_mean_results(
                performance_ensemble_user_list
            )
        }

        return results
    
    def run_our_method_varied_labeled_data(self):
        user_task, _ = self.benchmark.get_user_task_dataset_by_id(0, n_mix=2)
        X_train, _, _, _ = (
            user_task["X_train"],
            user_task["y_train"],
            user_task["X_test"],
            user_task["y_test"],
        )

        n_x_train = len(X_train)
        valid_sample_size_list = [i for i in n_sample_labeled_data_list if i < n_x_train]
        if self.task_name == "openml__pbc__4850":
            valid_sample_size_list = [10, 20, 50, 100, 200]

        results = {}
        for n_sample_labeled_data in valid_sample_size_list:
            Setup = copy.deepcopy(self.setup)
            Setup["User task parameters"]["n_sample_labeled_data"] = n_sample_labeled_data
            if n_sample_labeled_data >= 500:
                Setup["User task parameters"]["n_repeated"] = 3
            elif n_sample_labeled_data >= 1000:
                Setup["User task parameters"]["n_repeated"] = 1

            temp_results = self.run_our_method()

            results[n_sample_labeled_data] = {
                "Our method": temp_results["Our method"][0],
                "Self-training": temp_results["Self-training"][0],
                "Ensemble with user": temp_results["Ensemble with user"][0],
            }
        
        return results
    
    def aggregate_results(self, results_dict):
        aggregate_res_info(results_dict, self.res_dir)
    
    def plot_results(self, results_dict_varied_labeled_data):
        if self.task_type == "classification":
            plot_user_curve_aggregation_classification(results_dict_varied_labeled_data, self.res_dir)
            plot_all_classification_results(results_dict_varied_labeled_data, self.res_dir)
        elif self.task_type == "regression":
            plot_user_curve_aggregation_regression(results_dict_varied_labeled_data, self.res_dir)