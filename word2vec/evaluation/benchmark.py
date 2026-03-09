from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from loguru import logger

from word2vec.config.schema import Dataset, Word2VecConfig
from word2vec.dataset import get_raw_dataset
from word2vec.model import load_model_for_config
from word2vec.train import Word2VecModel


def eval(cfg: Word2VecConfig):
    dataset_to_eval_fn: Callable[[Word2VecModel], Any] = {
        Dataset.simlex999: eval_simlex
    }

    model = load_model_for_config(cfg)
    if model is None:
        raise RuntimeError("Train the model first or change the configuration.")

    return dataset_to_eval_fn[cfg.eval.dataset](model)


@dataclass
class SimlexResult:
    correlation: float
    coverage: float


def eval_simlex(model: Word2VecModel):
    dataset = cast(
        pd.DataFrame, get_raw_dataset(Dataset.simlex999)["train"].to_pandas()
    )
    present = np.ones(len(dataset), dtype=bool)
    sims = np.zeros(len(dataset))

    for i, row in dataset.iterrows():
        word_1, word_2 = row["word1"], row["word2"]
        if word_1 not in model.word_to_idx or word_2 not in model.word_to_idx:
            present[i] = False
            continue

        sims[i] = model.similarity(word_1, word_2)

    dataset_sims_filtered = dataset["similarity"].to_numpy()[present]
    model_sims_filtered = sims[present]

    df = pd.DataFrame(
        {
            "human": dataset_sims_filtered,
            "model": model_sims_filtered,
        }
    )

    res = SimlexResult(df["human"].corr(df["model"], method="spearman"), present.mean())
    logger.info(
        f"Simlex999 benchmark results: coverage={res.coverage}, corr={res.correlation}"
    )
