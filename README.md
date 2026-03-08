# word2vec

This is an implementation of the word2vec embedding calculation method in pure numpy, with manual gradient derivation. Done as a take-home for an internship application.

Instructions:
```
Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).
The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the code, gradient derivation, and possible alternative implementations or optimizations.
```


I tried to be as faithful to the [ original C implementation ](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c) as I could.

What's implemented:
- skip-gram with negative sampling over CBOW since it allegedly has nicer gradient flow
- subsampling with a precalculated table as a more efficient alternative to the original implementation

## Resources

Here's what I used to study the subject and implement this:
- [the OG paper](https://arxiv.org/pdf/1301.3781)
- [extensions](https://arxiv.org/pdf/1310.4546)
- [the original C word2vec impl with comments](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c)

## Repository structure
```
.
├── main.py                                 # Main entrypoint
├── word2vec
│   ├── train.py                            # Training loop
│   ├── dataset.py                          # Preprocessing
│   └── config
│       └── schema.py                       # Dataclass Hydra config schema
│
├── conf
│   └── config.yaml                         # Hydra configuration
│
├── data
│   ├── raw                                 # Raw datasets
│   └── processed                           # Preprocessed datasets
│
├── models                                  # Trained pickled models
│
├── derivation.{typ,pdf}                    # Gradient derivations
└── Justfile                                # Quick commands to run scripts
```


# TODO

### Determine testing strategy
- word similarity datasets
    - wordsim353
    - simlex999
    - spearman correlation between cosine and human scores
- word analogy task [ google analogy test set ](https://www.aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art))
- knn query
- possibly train some sort of classifier on top of this

