import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

from word2vec.config import MODELS_DIR
from word2vec.config.schema import Word2VecConfig
from word2vec.dataset import preprocess_dataset


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def path_for_model_config(cfg: Word2VecConfig):
    return (
        MODELS_DIR
        / f"{cfg.dataset}_d{cfg.training.latent_dimensionality}_k{cfg.training.num_negative_samples}_e{cfg.training.num_epochs}.pkl"
    )


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


def train_or_load(cfg: Word2VecConfig):
    path = path_for_model_config(cfg)
    if path.exists() and not cfg.training.force_train:
        logger.info(f"Model found at {path}, loading from disk")
        return Word2VecModel.from_file(path)

    logger.info("Training model...")
    model = training_loop(cfg)
    model.save(path)
    logger.info(f"Model saved to {path}")
    return model


def training_loop(cfg: Word2VecConfig):
    dataset = preprocess_dataset(cfg)
    rng = np.random.default_rng(cfg.training.seed)
    d = cfg.training.latent_dimensionality
    vocab_size = len(dataset.vocab)

    # https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c#L773
    W_in = rng.uniform(-0.5 / d, 0.5 / d, vocab_size * d).reshape((vocab_size, d))
    W_out = np.zeros((vocab_size, d))

    epochs = cfg.training.num_epochs
    lr_start = cfg.training.lr_start
    n_negative_samples = cfg.training.num_negative_samples
    max_window_size = cfg.training.max_neighbourhood_size
    total_tokens = dataset.total_tokens
    unigram_table = dataset.unigram_table

    for epoch in tqdm(range(epochs), "Epochs"):
        for center_corpus_idx, center_vocab_idx in enumerate(dataset.corpus):
            # https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c#L840C6-L846C6
            progress = center_corpus_idx / (epochs * total_tokens)
            lr = lr_start * (1 - progress)
            lr = max(lr, 0.0001 * lr_start)

            if center_corpus_idx % 10000 == 0:
                logger.info(
                    f"Epoch {epoch}, token {center_corpus_idx}/{total_tokens}, lr {lr:.6f}"
                )

            # random window size
            window = rng.integers(1, max_window_size + 1)

            # get context words
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue

                context_idx = center_corpus_idx + offset
                if context_idx < 0 or context_idx >= total_tokens:
                    continue

                context_vocab_idx = dataset.corpus[context_idx]
                negative_samples = unigram_table[
                    rng.integers(0, len(unigram_table), n_negative_samples)
                ]

                # TODO: subsampling
                # TODO: update derivation doc to derive for neg log

                # positive example update
                v_in = W_in[center_vocab_idx].copy()
                v_out = W_out[context_vocab_idx]
                score_pos = v_in @ v_out
                g_pos = sigmoid(score_pos) - 1

                W_in[center_vocab_idx] -= lr * g_pos * v_out
                W_out[context_vocab_idx] -= lr * g_pos * v_in

                # negative examples
                v_out_neg = W_out[negative_samples]  # (k, d)
                scores_neg = v_out_neg @ v_in  # (k,)
                g_neg = sigmoid(scores_neg)  # (k, )

                # lr * sum(g_neg[i] * v_out_neg[i]) for i in range(k))
                W_in[center_vocab_idx] -= lr * (g_neg @ v_out_neg)
                # g_neg[i] * v_in for i in range(k))
                W_out[negative_samples] -= lr * np.outer(g_neg, v_in)

    return Word2VecModel(
        vocab=dataset.vocab,
        word_to_idx=dataset.word_to_idx,
        embeddings=W_in,
    )
