# import the necessary packages
import torch
import os

# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = "dataset"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output model, label encoder, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pth"])
LE_PATH = os.path.sep.join([BASE_OUTPUT, "le.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# determine the current device and based on that set the pin memory
# flag
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

# TODO: for my dataset (may change overtime)
# MEAN = [0.3469, 0.3043, 0.2974]
# STD = [0.3204, 0.3003, 0.2957]
MEAN = [0.3649, 0.3242, 0.3095]
STD = [0.3182, 0.3016, 0.2971]
# initialize our initial learning rate, number of epochs to train
# for, and the batch size
INIT_LR = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 4

# specify the loss weights
# LABELS = 1.0
BBOX = 1.0