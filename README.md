# Predicting the Helpfulness of Amazon User Reviews using Machine Learning and Natural Language Processing

User-generated online reviews are important because they can influence purchasing decisions. On Amazon, a user may rate a review as either helpful or unhelpful. By default, Amazon sorts reviews by helpfulness and displays the most helpful reviews first. A machine learning model that predicts the helpfulness of new reviews could be a useful way of categorizing reviews that have not yet amassed helpfulness ratings.

The hypothesis is that Amazon product reviews which users consider helpful share common traits that can be determined from the review text. The hypothesis is tested by training models on different sets of features and seeing which features are most predictive of review helpfulness. We train logistic regression, Naive Bayes, and SVM models on groups of features (structural, part-of-speech, emotionality, and word frequencies) extracted from Amazon reviews.

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

- `path`: reviews path (directory or file in .json.gz format)
- `test_path`: test reviews path (optional)

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
