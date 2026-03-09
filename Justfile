# initialize the venv
init:
  uv sync

# Preprocess the dataset
preprocess:
  uv run -m main stage=preprocess

# Train a model
train:
  uv run -m main stage=train training.force_train=True

# Run demo knn queries
knn:
  uv run -m main stage=knn

# Demo of embedding arithmetic, e.g. smallest-small+big=biggest
arithmetic:
  uv run -m main stage=arithmetic

# Benchmark a model on simlex-999
benchmark:
  uv run -m main stage=benchmark

types:
  uvx ty check

lint:
  ruff check
