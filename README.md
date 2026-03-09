# word2vec

This is an implementation of the word2vec embedding calculation method in pure numpy, with manual gradient derivation. Done as a take-home for an internship application.

Instructions:
```
Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).
The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the code, gradient derivation, and possible alternative implementations or optimizations.
```


I tried to be as faithful to the [ original C implementation ](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c) as I could.

What's implemented:
- skip-gram with negative sampling over CBOW since it allegedly has nicer gradient flow than CBOW
- subsampling with a precalculated table as a more efficient alternative to the original implementation

## Resources

Here's what I used to study the subject and implement this:
- [the OG paper](https://arxiv.org/pdf/1301.3781)
- [extensions](https://arxiv.org/pdf/1310.4546)
- [the original C word2vec impl with comments](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c)

## Repository structure
```
.
├── word2vec
│   ├── model.py                            # Model wrapper
│   ├── train.py                            # Training loop
│   ├── dataset.py                          # Preprocessing
│   ├── evaluation
│   │   ├── benchmark.py                    # SimLex benchmark
│   │   ├── knn.py                          # KNN query demo
│   │   └── arithmetic.py                   # Embedding arithmetics demo
│   └── config
│       └── schema.py                       # Dataclass Hydra config schema
│
├────── notebooks
│       └── demo.ipynb                      # Notebook with a quick demo of a trained model
│
├── derivation.{pdf,typ}                          # Gradient derivations
│
├── data
│   ├── raw                                 # Raw datasets
│   └── processed                           # Processed datasets
│
├── models                                  # Trained pickled models
│
├── conf
│   └── config.yaml                         # Hydra configuration
│
├── main.py                                 # Main entrypoint
└── Justfile                                # Quick commands to run scripts
```

Here are all the configuration options explained:

```
dataset: text8                     # training corpus (cleaned 17M-token Wikipedia dump)
stage: preprocess                  # pipeline stage to run first

preprocessing:
  min_token_corpus_count: 5        # drop tokens that appear fewer times than this in the corpus
  unigram_table_size: 100_000_000  # size of the lookup table used to efficiently sample negatives
  neg_sampling_dist_exponent: 0.75 # exponent applied to token frequencies when building the negative sampling distribution
  force_preprocess: false          # recompute preprocessing even if a cached processed dataset exists
  subsampling_threshold: 1e-5      # threshold controlling probability of discarding very frequent words

training:
  max_neighbourhood_size: 5        # maximum context window radius around each center word
  latent_dimensionality: 300       # dimensionality of the learned word embeddings
  num_negative_samples: 10         # number of negative examples drawn per positive training pair
  lr_start: 0.025                  # initial SGD learning rate (decays linearly during training)
  num_epochs: 5                    # number of passes over the corpus
  seed: 42                         # RNG seed for reproducibility
  force_train: false               # retrain even if a model with this config already exists

benchmark:
  dataset: simlex999               # evaluation dataset used for computing similarity correlation; only simlex is supported

knn:
  words:                           # example words for nearest-neighbor inspection
    - word1
    - word2

arithmetic:
  patterns:                        # word analogy tests of the form a - b + c ≈ d
    - king,man,woman,queen
```


Every hyperparameter is configurable in `config.yaml`. If you want to play around with a pretrained model, you can pull the benchmarked below instance using git lfs. Every script from `evaluation/` tries to load a model that conforms to the configuration from config.yaml and fails if it's not in `models/`. If you want to play around with the params and train your instance, customize the yaml and run `just train`.

There is a single entrypoint to the repo, the actual script/stage to run is parametrized with the `stage` parameter, which you can override via the command line. However, it's simpler to use the predefined Justfile commands (see `just --list`). Note that each script has corresponding settings in `config.yaml` that you might want to tweak.

## Results

### SimLex-999 benchmark

A model trained on the [text8](https://huggingface.co/roshbeed/text8-dataset) dataset (~170mln words) with the following hyperparameter setup:
```
max_neighbourhood_size: 5
latent_dimensionality: 300
num_negative_samples: 10
lr_start: 0.025
num_epochs: 5
```

achieves the Spearman correlation coefficient of 0.3 on the [simlex-999](https://fh295.github.io/simlex.html) dataset, which sounds about right compared to what the website says - they state that a negative sampling skip-gram implementation trained on 1bn tokens achieved 0.37.

### KNN queries

Results of a few knn queries:
```
Pasta:
1. savoury: 0.8694
2. sausage: 0.8567
3. desserts: 0.8557
4. veal: 0.8530
5. liqueur: 0.8499

Four:
1. three: 0.9424
2. six: 0.9397
3. five: 0.9372
4. seven: 0.9277
5. eight: 0.9147

Ronald:
1. reagan: 0.7949
2. seretse: 0.7514
3. mondale: 0.7510
4. mcnamara: 0.7500
5. agnew: 0.7500

Fruit:
1. fruits: 0.8454
2. vegetables: 0.8264
3. juices: 0.8150
4. vegetable: 0.7964
5. rotting: 0.7927

Guitar:
1. bass: 0.8140
2. guitars: 0.7853
3. harmonica: 0.7806
4. saxophone: 0.7734
5. clavichord: 0.7625

Happiness:
1. goodness: 0.7697
2. toil: 0.7658
3. kindness: 0.7630
4. infirmities: 0.7599
5. unhappiness: 0.7582
```

They mostly seem to make some semantic sense.

### Attempts at finding some deeper arithmetic relationships

I also tried to recreate the `biggest-big+small=smallest` example from the original paper, as well as a few different ones, here are the results:

```
Query: teacher - school + hospital = ? (Expected: doctor/nurse)
  hospital: 0.7443
  teacher: 0.7136
  nurse: 0.6606
  housekeeper: 0.6191
  schoolteacher: 0.6124

Query: biggest - big + small = ? (Expected: smallest)
  small: 0.7250
  biggest: 0.7194
  largest: 0.6700
  large: 0.6255
  poorest: 0.6005

Query: cat - kitten + puppy = ? (Expected: dog)
  cat: 0.9171
  puppy: 0.7461
  eared: 0.7281
  guanaco: 0.7218
  rattus: 0.7042

Query: banana - yellow + red = ? (Expected: apple)
  banana: 0.8308
  red: 0.6301
  daiquiri: 0.6059
  iced: 0.5994
  buna: 0.5969
```

These didn't come out too good, at least the nurse example stuck, so there's some semblance of well trained embeddings. I would probably need a larger corpus and a faster CPU to improve these results.

## Possible future optimizations

- testing out whether averaging `W_in` and `W_out` would give better results; in the original implementation they opted not to do this
- experiment with weighting context words based on their distance 
