import os

# Base project directory (use abspath for portability on Render/Linux)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset path used by the hosted retriever.
DATASET_JSON_PATH = os.getenv("DATASET_JSON_PATH", os.path.join(BASE_DIR, "hotel_dataset.json"))
