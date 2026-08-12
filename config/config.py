"""
Master config module. Import `from config.config import CFG` anywhere
in the project to get a single namespace with every sub-config.
"""

from types import SimpleNamespace

from config import paths
from config.model_config import (
    DATA_CONFIG, PATCHTST_CONFIG, EXPERT_CONFIG, QUANTILE_CONFIG, EMBEDDING_CONFIG,
)
from config.training_config import SPLIT_CONFIG, TRAINING_CONFIG
from config.logging_config import LOG_LEVEL, LOG_FORMAT, DATE_FORMAT, LOG_FILE_NAME

CFG = SimpleNamespace(
    paths=paths,
    data=DATA_CONFIG,
    patchtst=PATCHTST_CONFIG,
    experts=EXPERT_CONFIG,
    quantiles=QUANTILE_CONFIG,
    embedding=EMBEDDING_CONFIG,
    split=SPLIT_CONFIG,
    training=TRAINING_CONFIG,
    log_level=LOG_LEVEL,
    log_format=LOG_FORMAT,
    date_format=DATE_FORMAT,
    log_file_name=LOG_FILE_NAME,
)
