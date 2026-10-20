from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = Path().resolve().parent

DATA_DIR = PROJECT_ROOT / 'data'
CV_DIR = DATA_DIR / 'cv'
JD_DIR = DATA_DIR / 'jd'

OUTPUT_DIR = PROJECT_ROOT / 'output'