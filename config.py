import os

# Get the absolute path of the project's root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

# Define paths relative to BASE_DIR
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Example file paths
TRAIN_CAPTIONS = os.path.join(DATA_DIR, "Captions", "Train.jsonl")
TEST_CAPTIONS = os.path.join(DATA_DIR, "Captions", "Test.jsonl")
VALID_CAPTIONS = os.path.join(DATA_DIR, "Captions", "Valid.jsonl")
IMAGE_PATH = os.path.join(DATA_DIR, "images")
OUT_FILE = os.path.join(OUTPUTS_DIR, "output.pkl")
