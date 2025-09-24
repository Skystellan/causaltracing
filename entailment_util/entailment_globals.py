from pathlib import Path
import yaml

# Load entailment-specific configuration
with open("entailment_globals.yml", "r") as stream:
    data = yaml.safe_load(stream)

(
    ENTAILMENT_RESULTS_DIR,
    ENTAILMENT_DATA_DIR,
    ENTAILMENT_STATS_DIR,
    ENTAILMENT_HPARAMS_DIR,
    ENTAILMENT_CACHE_DIR,
    ENTAILMENT_VIZ_DIR,
) = (
    Path(z)
    for z in [
        data["ENTAILMENT_RESULTS_DIR"],
        data["ENTAILMENT_DATA_DIR"],
        data["ENTAILMENT_STATS_DIR"],
        data["ENTAILMENT_HPARAMS_DIR"],
        data["ENTAILMENT_CACHE_DIR"],
        data["ENTAILMENT_VIZ_DIR"],
    ]
)

# Alias for compatibility with ROME's DATA_DIR
DATA_DIR = ENTAILMENT_DATA_DIR

# Create directories if they don't exist
for directory in [
    ENTAILMENT_RESULTS_DIR,
    ENTAILMENT_DATA_DIR,
    ENTAILMENT_STATS_DIR,
    ENTAILMENT_HPARAMS_DIR,
    ENTAILMENT_CACHE_DIR,
    ENTAILMENT_VIZ_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)