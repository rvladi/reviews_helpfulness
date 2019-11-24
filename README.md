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

- unigram counts
    - all words
    - emotion words only
    - exclude emotion words
    - part of speech
- review length
- sentence lengths

### Packages

In your virtual environment, install `numpy`, `nltk`, and `scikit-learn`.

### Usage

```
python analyze.py <file_name>
```

`file_name`: reviews file name (.json.gz format)

### Example

```
python analyze.py reviews/reviews_Automotive_5.json.gz
```

**Note:** Download large datasets locally but don't push them to the repository.

### Results

- Logistic Regression: `log_reg.csv`
- Naive Bayes: `naive_bayes.csv`
