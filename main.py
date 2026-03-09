from typing import Any, Callable

import hydra

from word2vec.config.schema import Stage, Word2VecConfig, register_configs
from word2vec.dataset import preprocess_dataset
from word2vec.evaluation.arithmetic import test_arithmetic_operations
from word2vec.evaluation.benchmark import run_benchmark
from word2vec.evaluation.knn import knn_demo
from word2vec.train import train_or_load

register_configs()

stage_to_fn: dict[Stage, Callable[[Word2VecConfig], Any]] = {
    Stage.train: train_or_load,
    Stage.preprocess: preprocess_dataset,
    Stage.benchmark: run_benchmark,
    Stage.knn: knn_demo,
    Stage.arithmetic: test_arithmetic_operations,
}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Word2VecConfig):
    fn = stage_to_fn[cfg.stage]
    fn(cfg)


if __name__ == "__main__":
    main()
