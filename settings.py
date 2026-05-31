import os

# Base project directory (use abspath for portability on Render/Linux)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FAISS index path (on-disk directory). Defaults to a folder named `faiss_index` in the project root.
# You can override this by modifying this value or by importing and setting it at runtime.
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
