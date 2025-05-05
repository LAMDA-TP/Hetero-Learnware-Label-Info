import os

ROOT_PATH = os.path.abspath(os.path.join(__file__, ".."))
STORAGE_DIR = os.path.join(ROOT_PATH, "storage")

SPLIT_DATA_DIR = os.path.join(STORAGE_DIR, "split_data")
MODEL_DIR = os.path.join(STORAGE_DIR, "models")

LEARNWARE_DIR = os.path.join(STORAGE_DIR, "learnware_pool")
LEARNWARE_ZIP_DIR = os.path.join(LEARNWARE_DIR, "zips")
LEARNWARE_FOLDER_DIR = os.path.join(LEARNWARE_DIR, "learnwares")