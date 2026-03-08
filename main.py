from typing import Callable

import hydra
from loguru import logger
from omegaconf import OmegaConf

from word2vec.config.schema import Stage, Word2VecConfig, register_configs
from word2vec.dataset import preprocess_dataset
from word2vec.train import train

register_configs()

stage_to_fn: dict[Stage, Callable[[Word2VecConfig], None]] = {
    Stage.train: train,
    Stage.preprocess: preprocess_dataset,
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Word2VecConfig):
    fn = stage_to_fn[cfg.stage]
    logger.info(f"Running stage {cfg.stage} with config {OmegaConf.to_yaml(cfg)}")
    fn(cfg)


if __name__ == "__main__":
    main()
