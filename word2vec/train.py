import numpy as np
from loguru import logger
from tqdm import tqdm

from word2vec.config.schema import Word2VecConfig
from word2vec.dataset import preprocess_dataset
from word2vec.model import Word2VecModel, load_model_for_config, path_for_model_config


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def train_or_load(cfg: Word2VecConfig):
    if (model := load_model_for_config(cfg)) is not None:
        return model

    logger.info("Training model...")
    model = training_loop(cfg)
    path = path_for_model_config(cfg)
    model.save(path)
    logger.info(f"Model saved to {path}")
    return model


def training_loop(cfg: Word2VecConfig):
    dataset = preprocess_dataset(cfg)
    rng = np.random.default_rng(cfg.training.seed)
    d = cfg.training.latent_dimensionality
    vocab_size = len(dataset.vocab)

    # https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c#L773
    W_in = (
        rng.uniform(-0.5 / d, 0.5 / d, vocab_size * d)
        .reshape((vocab_size, d))
        .astype(np.float32)
    )
    W_out = np.zeros((vocab_size, d)).astype(np.float32)

    epochs = cfg.training.num_epochs
    lr_start = cfg.training.lr_start
    n_negative_samples = cfg.training.num_negative_samples
    max_window_size = cfg.training.max_neighbourhood_size
    total_tokens = dataset.total_tokens
    unigram_table = dataset.unigram_table
    subsampling_proba = dataset.subsampling_proba

    for epoch in tqdm(range(epochs), "Epochs"):
        for center_corpus_idx, center_vocab_idx in enumerate(dataset.corpus):
            # https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c#L840-L846
            progress = center_corpus_idx / (epochs * total_tokens)
            lr = lr_start * (1 - progress)
            lr = max(lr, 0.0001 * lr_start)

            if center_corpus_idx % 100_000 == 0:
                logger.info(
                    f"Epoch {epoch}, token {center_corpus_idx}/{total_tokens}, lr {lr:.6f}"
                )

            # https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c#L905
            # this should be at the start of the loop for max speedup, but it messes up my logs
            if rng.random() < subsampling_proba[center_vocab_idx]:
                continue

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
