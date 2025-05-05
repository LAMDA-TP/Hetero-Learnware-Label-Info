class ExperimentSetup:
    def __init__(self) -> None:
        self.learnware_args = {
            "n_splits": 4,
            "n_learnware_blocks": 3,
            "reduced_set_size": 50,
        }
        self.training_args = {
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "epochs": 300,  
            "architecture": {
                "hidden_dim": 32,
                "subspace_dim": 16,
            },
            "loss_weights": {
                "weight_cont": 1e2,  
                "weight_cont_supervised": 0,  
                "weight_rec": 1, 
                "weight_supervised": 1,  
                "weight_global": 0,  
                "use_weight": True,
            },
            "mmd_calculator_paras": {
                "num_bins": 10,
                "cond_mmd_coef": 1,
            },
            "rkme_for_training": "LabeledSpecificationCLS",
        }
        self.matching_args = {
            "gamma": 0.1,
            "num_bins": self.training_args["mmd_calculator_paras"]["num_bins"],
            "cond_mmd_coef": self.training_args["mmd_calculator_paras"][
                "cond_mmd_coef"
            ],
        }
        self.user_task_paras = {
            "n_mix_list": [2],
            "n_sample_labeled_data": 100,
            "user_reduced_set_size": 50,
            "reduce": True,
            "n_repeated": 5,
        }

    def _get_setup_paras(self):
        setup = {
            "Learnware arguments": self.learnware_args,
            "Training arguments": self.training_args,
            "Matching arguments": self.matching_args,
            "User task parameters": self.user_task_paras,
        }
        return setup

    def get_setup_paras(self, setup_name="classification"):
        if setup_name == "classification":
            setup = self._get_setup_paras()
            return setup
        elif setup_name == "regression":
            setup = self._get_setup_paras()
            setup["Training arguments"]["rkme_for_training"] = "LabeledSpecification"
            return setup
