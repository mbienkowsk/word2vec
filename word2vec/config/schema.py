from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore


class Dataset(str, Enum):
    text8 = "text8"


class Stage(str, Enum):
    download_dataset = "download_dataset"


@dataclass
class Word2VecConfig:
    dataset: Dataset
    stage: Stage


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Word2VecConfig)
