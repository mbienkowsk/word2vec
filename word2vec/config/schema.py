from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore


class Dataset(str, Enum):
    text8 = "text8"


class Stage(str, Enum):
    train = "train"
    preprocess = "preprocess"


@dataclass
class PreprocessingConfig:
    # exponent for the unigram distribution for negative sampling
    neg_sampling_dist_exponent: float

    # tokens with count < min_count get filtered out
    min_token_corpus_count: int

    # preprocess even if there is a processed copy in PROCESSED_DATA_DIR
    force_preprocess: bool


@dataclass
class TrainingConfig:
    max_neighbourhood_size: int
    latent_dimensionality: int
    num_negative_samples: int
    lr_start: float
    num_epochs: int
    seed: int
    subsampling_threshold: float
    force_train: bool


@dataclass
class Word2VecConfig:
    dataset: Dataset
    stage: Stage
    preprocessing: PreprocessingConfig
    training: TrainingConfig


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Word2VecConfig)
