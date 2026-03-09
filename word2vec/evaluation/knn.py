import sys

from loguru import logger

from word2vec.config.schema import Word2VecConfig
from word2vec.model import load_model_for_config


def knn_demo(cfg: Word2VecConfig):
    model = load_model_for_config(cfg)
    if model is None:
        logger.error("Model file missing, train it first or change the configuration.")
        sys.exit(1)

    for word in cfg.knn.words:
        knn_result = model.knn(word)
        print(f"\nTop 5 nearest neighbors for {word}:")
        for idx, (word, sim) in enumerate(knn_result, start=1):
            print(f"{idx}. {word}: {sim:.4f}")
