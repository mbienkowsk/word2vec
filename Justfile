init:
  uv sync

debug:
  uv run -m main stage=debug

train:
  uv run -m main stage=train training.force_train=True

init_train: init train
