import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from word2vec.config import MODELS_DIR
from word2vec.config.schema import Word2VecConfig


def path_for_model_config(cfg: Word2VecConfig):
    return (
        MODELS_DIR
        / f"{cfg.dataset}_d{cfg.training.latent_dimensionality}_k{cfg.training.num_negative_samples}_e{cfg.training.num_epochs}.pkl"
    )


def load_model_for_config(cfg: Word2VecConfig):
    if (path := path_for_model_config(cfg)).exists:
        logger.info(f"Model found at {path}, loading from disk")
        return Word2VecModel.from_file(path)

    logger.warning(f"Looked at {path}, didn't find the model.")
    return None


@dataclass
class Word2VecModel:
    vocab: list[str]
    word_to_idx: dict[str, int]
    embeddings: np.ndarray

    def embedding(self, word: str) -> np.ndarray:
        idx = self.word_to_idx[word]
        return self.embeddings[idx]

    def similarity(self, word1: str, word2: str) -> float:
        emb1 = self.embedding(word1)
        emb2 = self.embedding(word2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    @staticmethod
    def from_file(path: Path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
