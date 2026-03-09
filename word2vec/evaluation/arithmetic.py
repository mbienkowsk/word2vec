import numpy as np

from word2vec.config.schema import Word2VecConfig
from word2vec.model import load_model_for_config


def test_arithmetic_operations(cfg: Word2VecConfig):
    model = load_model_for_config(cfg)
    assert model is not None

    for pattern in cfg.arithmetic.patterns:
        try:
            a, b, c, d = [w.strip() for w in pattern.split(",")]
        except ValueError:
            print(f"Pattern '{pattern}' is invalid, skipping.")
            continue

        emb_a, emb_b, emb_c = (
            model.embedding(a),
            model.embedding(b),
            model.embedding(c),
        )
        result_emb = emb_a - emb_b + emb_c
        result_emb /= np.linalg.norm(result_emb)

        matches = model.knn_for_emb(result_emb, k=5)

        print(f"\nPattern: {a} - {b} + {c} = ? (Expected: {d})")
        for word, score in matches:
            print(f"  {word}: {score:.4f}")
