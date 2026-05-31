import os

# Base project directory (use abspath for portability on Render/Linux)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FAISS index path (on-disk directory). Defaults to a folder named `faiss_index` in the project root.
# On Render, keep this inside the app directory unless FAISS_INDEX_PATH is set explicitly.
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(BASE_DIR, "faiss_index"))

# Dataset path used for rebuilding the FAISS index if the saved files are missing.
DATASET_JSON_PATH = os.getenv("DATASET_JSON_PATH", os.path.join(BASE_DIR, "hotel_dataset.json"))

# Render sets this environment variable automatically.
IS_RENDER = bool(os.getenv("RENDER"))
