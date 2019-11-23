import argparse
import gzip
import json
import string
import time
from enum import Enum

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

# maximum number of reviews
MAX_REVIEWS = 10000

# number of sentences per review
NUM_SENTENCES = 100

# maximum number of iterations for Logistic Regression
MAX_ITERATIONS = 100

# punctuation table
punctuation_table = str.maketrans('', '', string.punctuation)

# Porter stemmer
stemmer = PorterStemmer()

# WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# stop words
stop_words = set(stopwords.words('english'))


def load_emotion_words(file_name):
    with open(file_name, 'r') as f:
        return set(stemmer.stem(line.strip()) for line in f)


# emotion words
emotion_words = load_emotion_words('emotions.txt')


class RunOption(Enum):
    ALL_WORDS = 1
    EMOTION_WORDS = 2
    EXCLUDE_EMOTION_WORDS = 3
    REVIEW_LENGTH = 4
    SENTENCE_LENGTH = 5


class Algo(Enum):
    LOG_REG = 1
    NAIVE_BAYES = 2


def stem(tokens):
    return [stemmer.stem(token) for token in tokens]


def lemmatize(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


def remove_stop_words(tokens):
    return [token for token in tokens if token not in stop_words]


def extract_emotion_words(tokens):
    return [token for token in tokens if token in emotion_words]


def exclude_emotion_words(tokens):
    return [token for token in tokens if token not in emotion_words]


def load_reviews(file_name):
    reviews = []

    with gzip.open(file_name, mode='r') as f:
        for count, line in enumerate(f):
            if count == MAX_REVIEWS:
                break
            entry = json.loads(line)
            review = entry['reviewText']
            helpful = entry['helpful']
            helpful_votes = helpful[0]
            total_votes = helpful[1]
            if helpful_votes * 2 > total_votes > 1:
                helpfulness = 'helpful'
            else:
                helpfulness = 'not_helpful'
            reviews.append((review, helpfulness))

    return reviews


def preprocess(reviews, run_option):
    preprocessed_reviews = []

    for review in reviews:
        tokens = word_tokenize(review)

        # remove punctuation
        tokens = [w.translate(punctuation_table) for w in tokens]

        # remove non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha()]

        # lowercase
        tokens = [token.lower() for token in tokens]

        # remove stopwords
        tokens = remove_stop_words(tokens)

        # stem (or lemmatize)
        tokens = stem(tokens)  # tokens = lemmatize(tokens)

        # extract or exclude emotion words
        if run_option is RunOption.EMOTION_WORDS:
            tokens = extract_emotion_words(tokens)
        elif run_option is RunOption.EXCLUDE_EMOTION_WORDS:
            tokens = exclude_emotion_words(tokens)

        review = ' '.join(tokens)
        preprocessed_reviews.append(review)

    return preprocessed_reviews


def get_features_words(reviews, run_option):
    reviews = preprocess(reviews, run_option)
    vocabulary_size = len(set(token for review in reviews for token in word_tokenize(review)))

    rank_threshold = 100
    max_features = vocabulary_size - rank_threshold if vocabulary_size > rank_threshold else None
    vectorizer = CountVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(reviews)

    return features


def get_features_lengths(reviews, run_option):
    features = []

    for review in reviews:
        if run_option is RunOption.REVIEW_LENGTH:
            features.append([len(review)])
        else:
            slen = [0] * NUM_SENTENCES  # sentence lengths of the review
            sentences = sent_tokenize(review)[:NUM_SENTENCES]  # list of sentences
            slen[:len(sentences)] = (len(sentence) for sentence in sentences)
            features.append(slen)

    return features


def get_features(reviews, run_option):
    if run_option is RunOption.ALL_WORDS or \
            run_option is RunOption.EMOTION_WORDS or \
            run_option is RunOption.EXCLUDE_EMOTION_WORDS:
        features = get_features_words(reviews, run_option)
    else:
        features = get_features_lengths(reviews, run_option)

    return features


def get_data_sets(reviews, run_option):
    reviews_text, labels = map(list, zip(*reviews))

    # extract features
    features = get_features(reviews_text, run_option)

    # split reviews into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(features, labels, train_size=0.9, random_state=2797)

    return x_train, x_test, y_train, y_test


def logistic_regression(title, x_train, x_test, y_train, y_test):
    # create hyperparameter options
    c = np.logspace(-4, 4, num=10)
    penalty = ['l1', 'l2']
    hyperparameters = {'C': c, 'penalty': penalty}

    # create grid search using 5-fold cross-validation
    log_reg = LogisticRegression(solver='liblinear', max_iter=MAX_ITERATIONS)
    grid_obj = GridSearchCV(log_reg, hyperparameters, scoring='accuracy', cv=5, n_jobs=-1)
    grid_obj.fit(X=x_train, y=y_train)

    # retrieve the best classifier
    classifier = grid_obj.best_estimator_

    # accuracy score on the dev sets
    accuracy = grid_obj.best_score_
    print('[{}] Logistic Regression accuracy: {:.1f}%'.format(title, accuracy * 100))

    return {'clf': classifier, 'x_test': x_test, 'y_test': y_test}


def naive_bayes(title, x_train, x_test, y_train, y_test):
    # train the model using the training set
    classifier = MultinomialNB()
    classifier.fit(x_train, y_train)

    # 5-fold cross-validation
    scores = cross_val_score(classifier, x_train, y_train, scoring='accuracy', cv=5, n_jobs=-1)

    # accuracy score on the dev sets
    accuracy = np.mean(scores)
    print('[{}] Naive Bayes accuracy: {:.1f}%'.format(title, accuracy * 100))

    return {'clf': classifier, 'x_test': x_test, 'y_test': y_test}


def print_results(title, model_accuracy, algo):
    h1 = '=' * 94
    h2 = '-' * 94

    print('\n\n{:^94}'.format(file_name + ' (' + title + ')'))
    print(h1)
    print('{:^1}{:^92}{:^1}'.format('|', 'Accuracy %', '|'))
    print(h2)
    print('{:^1}{:^16}{:^3}{:^16}{:^3}{:^16}{:^3}{:^16}{:^3}{:^16}{:^1}'.format('|', 'all words',
                                                                                '|', 'emotions only',
                                                                                '|', 'exclude emotions',
                                                                                '|', 'review length',
                                                                                '|', 'sentence length', '|'))
    print(h1)

    fmt = '{:^1}{:^16.1f}{:^3}{:^16.1f}{:^3}{:^16.1f}{:^3}{:^16.1f}{:^3}{:^16.1f}{:^1}'
    print(fmt.format('|', model_accuracy[RunOption.ALL_WORDS][algo] * 100, '|',
                     model_accuracy[RunOption.EMOTION_WORDS][algo] * 100, '|',
                     model_accuracy[RunOption.EXCLUDE_EMOTION_WORDS][algo] * 100, '|',
                     model_accuracy[RunOption.REVIEW_LENGTH][algo] * 100, '|',
                     model_accuracy[RunOption.SENTENCE_LENGTH][algo] * 100, '|', ))
    print(h1)


def save_results(file_name, category, model_accuracy, algo):
    entry = '{},{:^4.1f},{:^4.1f},{:^4.1f},{:^4.1f},{:^4.1f}\n'.format(
        category,
        model_accuracy[RunOption.ALL_WORDS][algo] * 100,
        model_accuracy[RunOption.EMOTION_WORDS][algo] * 100,
        model_accuracy[RunOption.EXCLUDE_EMOTION_WORDS][algo] * 100,
        model_accuracy[RunOption.REVIEW_LENGTH][algo] * 100,
        model_accuracy[RunOption.SENTENCE_LENGTH][algo] * 100)

    with open(file_name, mode='a+') as f:
        f.seek(0)
        lines = f.readlines()[1:]

        for i in range(len(lines)):
            if lines[i].startswith(category + ','):
                lines[i] = entry
                break
        else:
            lines.append(entry)
            lines.sort()

        f.seek(0)
        f.truncate()
        f.write('category,all words,emotions only,exclude emotions,review length,sentence length\n')
        f.writelines(lines)


def build_model(reviews, title, run_option):
    print('\n[' + title + '] Building models...')
    results = {
        Algo.LOG_REG: {'clf': None, 'x_test': None, 'y_test': None},
        Algo.NAIVE_BAYES: {'clf': None, 'x_test': None, 'y_test': None},
    }

    x_train, x_test, y_train, y_test = get_data_sets(reviews, run_option)

    print('[' + title + '] Running Logistic Regression...')
    results[Algo.LOG_REG] = logistic_regression(title, x_train, x_test, y_train, y_test)

    print('[' + title + '] Running Naive Bayes...')
    results[Algo.NAIVE_BAYES] = naive_bayes(title, x_train, x_test, y_train, y_test)

    return results


def calculate_accuracy(results):
    accuracy = {}

    for k, d in results.items():
        # retrieve the best classifier
        classifier = d['clf']
        x_test = d['x_test']
        y_test = d['y_test']

        # predict the labels for the test set
        y_predict = classifier.predict(x_test)

        # accuracy score on the test set
        accuracy[k] = metrics.accuracy_score(y_test, y_predict)

    return accuracy


# start time
start = time.perf_counter()

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('file_name', type=str, help='reviews file name')
args = parser.parse_args()
file_name = args.file_name
print('\nReviews file name: ' + file_name)

# load reviews [(review, helpfulness), ...]
print('\nLoading reviews...')
reviews = load_reviews(file_name)

# build models
results = {
    RunOption.ALL_WORDS: build_model(reviews, 'all words', RunOption.ALL_WORDS),
    RunOption.EMOTION_WORDS: build_model(reviews, 'emotion words only', RunOption.EMOTION_WORDS),
    RunOption.EXCLUDE_EMOTION_WORDS: build_model(reviews, 'exclude emotion words', RunOption.EXCLUDE_EMOTION_WORDS),
    RunOption.REVIEW_LENGTH: build_model(reviews, 'review length', RunOption.REVIEW_LENGTH),
    RunOption.SENTENCE_LENGTH: build_model(reviews, 'sentence length', RunOption.SENTENCE_LENGTH),
}

# calculate accuracy
model_accuracy = {
    RunOption.ALL_WORDS: calculate_accuracy(results[RunOption.ALL_WORDS]),
    RunOption.EMOTION_WORDS: calculate_accuracy(results[RunOption.EMOTION_WORDS]),
    RunOption.EXCLUDE_EMOTION_WORDS: calculate_accuracy(results[RunOption.EXCLUDE_EMOTION_WORDS]),
    RunOption.REVIEW_LENGTH: calculate_accuracy(results[RunOption.REVIEW_LENGTH]),
    RunOption.SENTENCE_LENGTH: calculate_accuracy(results[RunOption.SENTENCE_LENGTH]),
}

# print results
print_results('Logistic Regression', model_accuracy, Algo.LOG_REG)
print_results('Naive Bayes', model_accuracy, Algo.NAIVE_BAYES)

# save results
start_idx = file_name.find('_')
end_idx = file_name.find('_5')
category = file_name[start_idx + 1:end_idx].replace('_', ' ')
save_results('log_reg.csv', category, model_accuracy, Algo.LOG_REG)
save_results('naive_bayes.csv', category, model_accuracy, Algo.NAIVE_BAYES)

# elapsed time
print('\n\nElapsed time:', round(time.perf_counter() - start), 's')
