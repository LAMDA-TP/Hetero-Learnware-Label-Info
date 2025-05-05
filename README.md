# Handling Learnwares from Heterogeneous Feature Spaces with Explicit Label Exploitation<!-- omit in toc -->

This is the official implementation of the paper "Handling Learnwares from Heterogeneous Feature Spaces with Explicit Label Exploitation".



# The main results

After setting up the environment, navigate to the parent directory of the repository, then use the following command to reproduce the main results presented in Table 1 and Table 2, as well as the results on users with varying sizes of labeled data (Figures 4 and 5):

```
python -m Hetero-learnware-label-info.main --task_type <task_type>
```
The task_type argument should be set to either "classification" or "regression".


# Set up the environment

### Requirements

To reproduce the experiments, you need to first install the necessary dependencies with the following command:

```
pip install -r requirements.txt
```


### Data Preparation

Our code is tested on classification and regression tasks from the [TabZilla Benchmark](https://github.com/naszilla/tabzilla), which sources datasets from [OpenML](https://www.openml.org/).
To download and preprocess all datasets, follow the instructions in the TabZilla Benchmark's [Datasets section](https://github.com/naszilla/tabzilla?tab=readme-ov-file#datasets). This should create a local directory `TabZilla/datasets`. Navigate to the repository root and use the following command to finish data preparation:

```
mv TabZilla/datasets Hetero-learnware-label-info/benchmarks/tabzilla/datasets
```


# Code Overview
- `core`: Contains the implementation of the overall procedure of constructing and exploiting a heterogeneous learnware doc system.
    - `specification`: Generates model specifications and user requirements.
        - `labeled_specification.py`: Implements the unified specification for both classification and regression tasks.
        - `labeled_specification_cls.py`: Implements the specialized specification for classification tasks.
        - `distance.py`: Calculates distance objectives during specification generation.
    - `subspace` 
        - `model.py`: Implements the mapping functions.
        - `loss.py`: Implements the loss for subspace learning, including reconstruction_loss, supervised_loss and contrastive loss.
        - `utils.py`: Implements the training of the mapping functions.
    - `hetero_market.py`: Implements a `HeteroLearnwareMarket` which manages heterogeneous models based on the unified subspace and recommend learnwares based on user requirements.
    - `reuse.py`: Implements learnware reuse methods.
- `benchmarks`: Generate developer models and user tasks using the class `LearnwareBenchmark`.
    - `tabzilla`: Handles datasets from TabZilla benchmark customly using the class `TabzillaLoader`.