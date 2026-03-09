init:
  uv sync

preprocess:
  uv run -m main stage=preprocess

debug:
  uv run -m main stage=debug

train:
  uv run -m main stage=train training.force_train=True

eval:
  uv run -m main stage=eval

all: init preprocess train
