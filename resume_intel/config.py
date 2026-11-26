"""Global configuration. Every random choice in the project is derived from SEED."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"

SEED = 42
# Fixed reference date, so that "Present" in a resume resolves identically on every run.
REF_YEAR, REF_MONTH = 2026, 11

N_PER_CLASS = 250          # resumes per category (10 categories -> 2,500 resumes)
TEST_SIZE = 0.20
N_JOBS_PER_CLASS = 5       # job postings per category used for the ranking study
P_HYBRID = 0.15            # career-switcher resumes (text mixes two categories)
P_INFLATED = 0.15          # resumes with injected skill / experience inflation
P_NOISY = 0.20             # resumes containing typographical noise
