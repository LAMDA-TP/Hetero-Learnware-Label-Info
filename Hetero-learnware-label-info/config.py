import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(ROOT_PATH, "res")

for path in [RES_DIR]:
    if not os.path.exists(path):
        os.makedirs(path)

n_sample_labeled_data_list = [
    10,
    50,
    100,
    500,
    1000,
    2000,
    5000,
]

DATASET_LIST = {
    "classification": [
        "openml__credit-g__31",
        "openml__semeion__9964",
        "openml__mfeat-karhunen__16",
        "openml__splice__45",
        "openml__gina_agnostic__3891",
        "openml__Bioresponse__9910",
        "openml__sylvine__168912",
        "openml__christine__168908",
        "openml__first-order-theorem-proving__9985",
        "openml__satimage__2074",
        "openml__fabert__168910",
        "openml__GesturePhaseSegmentationProcessed__14969",
        "openml__robert__168332",
        "openml__artificial-characters__14964",
        "openml__eye_movements__3897",
        "openml__nursery__9892",
        "openml__eeg-eye-state__14951",
        "openml__magic__146206",
        "openml__riccardo__168338",
        "openml__guillermo__168337",
        "openml__nomao__9977",
        "openml__Click_prediction_small__190408",
        "openml__volkert__168331",
    ],
    "regression": [
        "openml__pbc__4850",
        "openml__colleges__359942",
        "openml__cpu_small__4883",
        "openml__kin8nm__2280",
        "openml__dataset_sales__190418",
        "openml__california__361089",
        "openml__aloi__12732",
    ],
}