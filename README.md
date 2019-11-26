# Reviews Helpfulness

## Reviews Dataset

http://jmcauley.ucsd.edu/data/amazon/

## Logistic Regression and Naive Bayes

### Preprocessing

- remove punctuation
- remove non-alphabetic tokens
- convert to lowercase
- remove stop words
- stem (or lemmatize)

### Features

- unigram frequencies (or counts)
    - all words
    - emotion words only ([Scherer, 2005](https://pdfs.semanticscholar.org/b8e6/98e8a7d968f3dba9040e2f5fafe7e5a2b095.pdf))
    - exclude emotion words
    - part of speech
- review length
- sentence lengths

### Packages

In your virtual environment, install `numpy`, `nltk`, and `scikit-learn`.

### Usage

```
python analyze.py <path> [<test_path>]
```

`path`: reviews path (directory or file in .json.gz format)
`test_path`: test reviews path (optional)

### Examples

```
# train and test the model on samples from all categories
python analyze.py reviews
```

```
# train and test the model on the "Automotive" category
python analyze.py reviews/reviews_Automotive_5.json.gz
```

```
# train the model on the "Video Games" category and test it on the "Electronics" category
python analyze.py reviews/reviews_Video_Games_5.json.gz reviews/reviews_Electronics_5.json.gz
```

**Note:** Download large datasets locally but don't push them to the repository.

### Results

- Logistic Regression: `log_reg.csv`
- Naive Bayes: `naive_bayes.csv`
