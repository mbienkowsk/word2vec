import hydra

from word2vec.config.schema import Word2VecConfig, register_configs

register_configs()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Word2VecConfig):
    print(cfg)


if __name__ == "__main__":
    main()
