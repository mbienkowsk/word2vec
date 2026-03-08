from word2vec.config.schema import Word2VecConfig
from word2vec.dataset import get_raw_dataset


def train(cfg: Word2VecConfig):
    dataset = get_raw_dataset(cfg.dataset)
    print(dataset)
