
from huggingface_hub import hf_hub_download

from word2vec.config import RAW_DATA_PATH
from word2vec.config.schema import Dataset


def get_raw_dataset(dataset: Dataset):
    match dataset:
        case Dataset.text8:
            path = hf_hub_download(
                repo_id="roshbeed/text8-dataset",
                filename="text8_full.txt",
                cache_dir=RAW_DATA_PATH,
            )

    with open(path, "r") as f:
        return f.read()
