from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore


class Dataset(str, Enum):
    text8 = "text8"
    simlex999 = "simlex999"


class Stage(str, Enum):
    train = "train"
    preprocess = "preprocess"
    eval = "eval"


@dataclass
class PreprocessingConfig:
    # exponent for the unigram distribution for negative sampling
    neg_sampling_dist_exponent: float

    # tokens with count < min_count get filtered out
    min_token_corpus_count: int

    # preprocess even if there is a processed copy in PROCESSED_DATA_DIR
    force_preprocess: bool

    # size of the unigram table for negative sampling
    # https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c#L133
    unigram_table_size: int

    # while this is technically a training hyperparam, it's used to calculate the proba table for subsampling, so it makes more sense to put it here
    subsampling_threshold: float


@dataclass
class EvalConfig:
    # even though we can't use every dataset, e.g. text8 to eval, this is just simpler
    dataset: Dataset


@dataclass
class TrainingConfig:
    max_neighbourhood_size: int
    latent_dimensionality: int
    num_negative_samples: int
    lr_start: float
    num_epochs: int
    seed: int
    force_train: bool


@dataclass
class Word2VecConfig:
    dataset: Dataset
    stage: Stage
    preprocessing: PreprocessingConfig
    training: TrainingConfig
    eval: EvalConfig


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Word2VecConfig)
