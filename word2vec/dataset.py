import pickle
from collections import Counter
from dataclasses import dataclass
from typing import cast

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from loguru import logger

from word2vec.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from word2vec.config.schema import Dataset, Word2VecConfig


def get_raw_dataset(dataset: Dataset):
    match dataset:
        case Dataset.text8:
            with open(
                hf_hub_download(
                    repo_id="roshbeed/text8-dataset",
                    filename="text8_full.txt",
                    cache_dir=RAW_DATA_DIR,
                ),
                "r",
            ) as f:
                return f.read()
        case Dataset.simlex999:
            return load_dataset("Yuti/SimLex-999")


@dataclass
class ProcessedDataset:
    corpus: np.ndarray
    word_counts: dict[str, int]
    vocab: list[str]
    unigram_table: np.ndarray
    total_tokens: int
    word_to_idx: dict[str, int]
    subsampling_proba: np.ndarray


def processed_dataset_path(dataset: Dataset):
    return PROCESSED_DATA_DIR / f"{dataset}.pkl"


def preprocess_dataset(cfg: Word2VecConfig) -> ProcessedDataset:
    if (
        path := processed_dataset_path(cfg.dataset)
    ).exists() and not cfg.preprocessing.force_preprocess:
        logger.info(f"Processed dataset {cfg.dataset} found, loading from disk")
        with open(path, "rb") as f:
            return cast(ProcessedDataset, pickle.load(f))

    raw = get_raw_dataset(cfg.dataset).split()

    # filter out rare words
    word_counts_raw = Counter(raw)

    logger.info(f"Vocab size before filtering: {len(word_counts_raw)}")
    logger.info(f"Number of words before filtering: {sum(word_counts_raw.values())}")

    word_counts_filtered = {
        word: count
        for word, count in word_counts_raw.items()
        if count >= cfg.preprocessing.min_token_corpus_count
    }

    logger.info(f"Vocab size after filtering: {len(word_counts_filtered)}")

    # rebuild corpus & vocab
    processed_vocab = list(word_counts_filtered.keys())
    word_to_idx = {word: idx for idx, word in enumerate(processed_vocab)}
    processed_corpus = np.array(
        [word_to_idx[word] for word in raw if word in word_counts_filtered],
        dtype=np.int32,
    )

    num_tokens = len(processed_corpus)
    assert num_tokens == sum(word_counts_filtered.values())
    assert len(processed_vocab) == len(word_counts_filtered)

    logger.info(f"Number of words after filtering: {num_tokens}")

    # compute negative sampling distribution
    counts = np.array(
        [word_counts_filtered[word] for word in processed_vocab], dtype=np.float32
    )
    weights = counts**cfg.preprocessing.neg_sampling_dist_exponent
    neg_sampling_distribution = weights / weights.sum()
    unigram_table = np.random.choice(
        len(processed_vocab),
        size=cfg.preprocessing.unigram_table_size,
        p=neg_sampling_distribution,
    ).astype(np.int32)

    dist = counts / counts.sum()
    subsampling_proba = np.maximum(
        0, 1 - np.sqrt(cfg.preprocessing.subsampling_threshold / dist)
    ).astype(np.float32)

    dataset = ProcessedDataset(
        processed_corpus,
        word_counts_filtered,
        processed_vocab,
        unigram_table,
        num_tokens,
        word_to_idx,
        subsampling_proba,
    )

    with open(processed_dataset_path(cfg.dataset), "wb") as f:
        pickle.dump(dataset, f)

    logger.info("Saved processed dataset to disk.")

    return dataset
