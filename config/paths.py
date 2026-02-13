from pathlib import Path
import os
import json

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent


# Other directories
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CONFIG_DIR = ROOT_DIR / "config"
SRC_DIR = ROOT_DIR / "src"

# Sub-directories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = DATA_DIR / "logs"

# Models sub-directories
SAVED_MODELS_DIR = MODELS_DIR / "saved"

# Settings file
SETTINGS_FILE = CONFIG_DIR / "settings.json"

MODEL_FILE_INDEX = json.load(open(SETTINGS_FILE)).get("model_file_index")
# Data dirs
TRAIN_FILE = RAW_DATA_DIR / f"train_FD00{MODEL_FILE_INDEX}.txt"
TEST_FILE = RAW_DATA_DIR / f"test_FD00{MODEL_FILE_INDEX}.txt"

# Config Yamls
AGENTS_FILE = CONFIG_DIR / "agents.yaml"
TASKS_FILE = CONFIG_DIR / "tasks.yaml"

# Trained models dirs
WATCHDOG_MODEL_PATH = SAVED_MODELS_DIR / "watchdog_model.pkl"
SCALER_PATH = SAVED_MODELS_DIR / "scaler.pkl"

# Log file path
DB_LOG_PATH = LOGS_DIR / "system_events.json"



def ensure_directories():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# runs at import
ensure_directories()