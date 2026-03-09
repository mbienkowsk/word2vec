import numpy as np

from word2vec.config.schema import Word2VecConfig
from word2vec.dataset import preprocess_dataset
from word2vec.model import load_model_for_config


def debug(cfg: Word2VecConfig):
    """This is a script for some debug prints, dilly-dallying, etc"""
    model = load_model_for_config(cfg)
    assert model is not None

    for word in cfg.debug.knn_words:
        print(f"\nTop 5 nearest neighbors for '{word}':")
        knn_result = model.knn(word)
        for idx, neighbor in enumerate(knn_result, 1):
            print(f"  {idx}. {neighbor}")

    emb_norms = np.linalg.norm(model.embeddings, axis=1)
    print(
        f"\nEmbedding norms: mean={emb_norms.mean():.4f}, std={emb_norms.std():.4f}, min={emb_norms.min():.4f}, max={emb_norms.max():.4f}"
    )

    dataset = preprocess_dataset(cfg)
    subsampling = dataset.subsampling_proba
    print(
        f"\nSubsampling probabilities: mean={subsampling.mean():.4f}, min={subsampling.min():.4f}, max={subsampling.max():.4f}, median={np.median(subsampling):.4f}"
    )

    print(np.mean(np.linalg.norm(model.emb_norm, axis=1)))
