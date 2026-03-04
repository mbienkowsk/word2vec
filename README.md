Implement the core training loop of word2vec in pure NumPy (no PyTorch / TensorFlow or other ML frameworks). The applicant is free to choose any suitable text dataset. The task is to implement the optimization procedure (forward pass, loss, gradients, and parameter updates) for a standard word2vec variant (e.g. skip-gram with negative sampling or CBOW).
The submitted solution should be fully understood by the applicant: during follow-up we will ask questions about the code, gradient derivation, and possible alternative implementations or optimizations.
Preferably, solutions should be provided as a link to a public GitHub repository.

Base paper:
https://arxiv.org/pdf/1301.3781
optimizations:
https://arxiv.org/pdf/1310.4546
unigram distribution modeling:
https://arxiv.org/pdf/2106.02289

Atomic tasks
1. Determine training dataset

2. Determine testing strategy
- word similarity datasets
    - wordsim353
    - simlex999
    - spearman correlation between cosine and human scores
- word analogy task [ google analogy test set ](https://www.aclweb.org/aclwiki/Google_analogy_test_set_(State_of_the_art))
- knn query
- possibly train some sort of classifier on top of this

3. Determine what to parametrize
- neighbourhood size
- latent dimensionality ~ 50-1000
- number of negative examples per position
- optimizer learning rate

4. Determine CBOW/Skip-Gram
- skipgram with negative sampling, remember to raise to 3/4

5. Determine high level architecture of the solution
    - gradients
    - how exactly does the network work, just 2 matrices + softmax?
6. Optimizations
    - subsampling

7. design choices
- batching?
- optimizer? gotta implement from scratch -> SGD

---

1. Preprocessing
- tokenize (clean punctuation, lowercase)
- count word frequencies
- remove words with count < min_count
- rebuild corpus
- build vocab mappings
- build & save unigram dist
