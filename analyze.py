import argparse
import gzip
import json
import os
import string
import time
from enum import Enum

import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

# N-gram frequencies (versus counts)
NGRAM_FREQUENCIES = True

# run Naive Bayes
RUN_NAIVE_BAYES = True

# run Logistic Regression
RUN_LOG_REG = True

# maximum number of iterations for Logistic Regression
MAX_ITERATIONS = 100

# maximum number of reviews
MAX_REVIEWS = 10000

# number of sentences per review
NUM_SENTENCES = 100

# punctuation table
punctuation_table = str.maketrans('', '', string.punctuation)

# Porter stemmer
stemmer = PorterStemmer()

# WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# stop words
stop_words = set(stopwords.words('english'))


def load_emotion_words(file_name):
    with open(file_name, mode='r') as f:
        return set(stemmer.stem(line.strip()) for line in f)


# emotion words
emotion_words = load_emotion_words('emotions.txt')


class RunOption(Enum):
    ALL_WORDS = 1
    EMOTION_WORDS = 2
    EXCLUDE_EMOTION_WORDS = 3
    REVIEW_LENGTH = 4
    SENTENCE_LENGTH = 5
    POS = 6


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


def load_reviews_from_file(file_name, reviews, max_reviews):
    total_helpful = 0
    total_not_helpful = 0

    with gzip.open(file_name, mode='r') as f:
        for count, line in enumerate(f):
            if count == max_reviews:
                break
            entry = json.loads(line)
            review = entry['reviewText']
            helpful = entry['helpful']
            helpful_votes = helpful[0]
            total_votes = helpful[1]
            if helpful_votes * 2 > total_votes > 1:
                helpfulness = 'helpful'
                total_helpful += 1
            else:
                helpfulness = 'not_helpful'
                total_not_helpful += 1
            reviews.append((review, helpfulness))

    return total_helpful, total_not_helpful


def load_reviews(path, reviews):
    total_helpful = 0
    total_not_helpful = 0

    if os.path.isdir(path):
        file_names = [path + os.sep + file_name for file_name in os.listdir(path) if file_name.endswith('.json.gz')]
        max_reviews = MAX_REVIEWS // len(file_names)
    else:
        file_names = [path]
        max_reviews = MAX_REVIEWS

    for file_name in file_names:
        helpful, not_helpful = load_reviews_from_file(file_name, reviews, max_reviews)
        total_helpful += helpful
        total_not_helpful += not_helpful

    total = total_helpful + total_not_helpful
    print('Total number of helpful reviews: {} ({:.1f}%)'.format(total_helpful, 100 * total_helpful / total))
    print('Total number of not helpful reviews: {} ({:.1f}%)'.format(total_not_helpful,
                                                                     100 * total_not_helpful / total))


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


def get_vectorizer(max_features=None):
    if NGRAM_FREQUENCIES:
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        vectorizer = CountVectorizer(max_features=max_features)
    return vectorizer


def get_features_words(reviews, run_option):
    reviews = preprocess(reviews, run_option)

    vocabulary_size = len(set(token for review in reviews for token in word_tokenize(review)))
    rank_threshold = 100
    max_features = vocabulary_size - rank_threshold if vocabulary_size > rank_threshold else None

    vectorizer = get_vectorizer(max_features)
    features = vectorizer.fit_transform(reviews)
    return features


def get_features_pos(reviews, run_option=None):
    reviews_tags = []

    for review in reviews:
        review_tags = []
        sentences = sent_tokenize(review)
        for sentence in sentences:
            word_tags = pos_tag(word_tokenize(sentence))
            _, tags = map(list, zip(*word_tags))
            review_tags.extend(tags)
        reviews_tags.append(' '.join(review_tags))

    vectorizer = get_vectorizer()
    features = vectorizer.fit_transform(reviews_tags)
    return features


def get_features_lengths(reviews, run_option):
    features = []

    for review in reviews:
        if run_option is RunOption.REVIEW_LENGTH:
            features.append([len(review)])
        else:  # run_option is RunOption.SENTENCE_LENGTH
            slen = [0] * NUM_SENTENCES  # sentence lengths of the review
            sentences = sent_tokenize(review)[:NUM_SENTENCES]  # list of sentences
            slen[:len(sentences)] = (len(sentence) for sentence in sentences)
            features.append(slen)

    return features


def get_features(reviews, run_option):
    options = {
        RunOption.ALL_WORDS: get_features_words,
        RunOption.EMOTION_WORDS: get_features_words,
        RunOption.EXCLUDE_EMOTION_WORDS: get_features_words,
        RunOption.REVIEW_LENGTH: get_features_lengths,
        RunOption.SENTENCE_LENGTH: get_features_lengths,
        RunOption.POS: get_features_pos,
    }

    features = options[run_option](reviews, run_option)
    return features


def get_data_sets(reviews, num_model_reviews, run_option):
    reviews_text, labels = map(list, zip(*reviews))

    # extract features
    features = get_features(reviews_text, run_option)

    # split reviews into train and test sets
    if num_model_reviews < len(reviews):
        x_train = features[:num_model_reviews]
        x_test = features[num_model_reviews:]
        y_train = labels[:num_model_reviews]
        y_test = labels[num_model_reviews:]
    else:
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


def print_results(title, path, test_path, model_accuracy, algo):
    h1 = '=' * 113
    h2 = '-' * 113

    print('\n\n{:^113}'.format(title))
    if test_path:
        print('{:^113}'.format('Model file name: ' + path))
        print('{:^113}'.format('Test file name: ' + test_path))
    else:
        print('{:^113}'.format('File name: ' + path))
    print(h1)
    print('{:^1}{:^111}{:^1}'.format('|', 'Accuracy %', '|'))
    print(h2)
    print('{:^1}{:^16}{:^3}{:^16}{:^3}{:^16}{:^3}{:^16}{:^3}{:^16}{:^3}{:^16}{:^1}'
          .format('|', 'all words',
                  '|', 'emotions only',
                  '|', 'exclude emotions',
                  '|', 'review length',
                  '|', 'sentence length',
                  '|', 'part of speech', '|'))
    print(h1)

    fmt = '{:^1}{:^16.1f}{:^3}{:^16.1f}{:^3}{:^16.1f}{:^3}{:^16.1f}{:^3}{:^16.1f}{:^3}{:^16.1f}{:^1}'
    print(fmt.format('|', model_accuracy[RunOption.ALL_WORDS][algo] * 100, '|',
                     model_accuracy[RunOption.EMOTION_WORDS][algo] * 100, '|',
                     model_accuracy[RunOption.EXCLUDE_EMOTION_WORDS][algo] * 100, '|',
                     model_accuracy[RunOption.REVIEW_LENGTH][algo] * 100, '|',
                     model_accuracy[RunOption.SENTENCE_LENGTH][algo] * 100, '|',
                     model_accuracy[RunOption.POS][algo] * 100, '|', ))
    print(h1)


def save_results(file_name, model_category, test_category, model_accuracy, algo):
    entry = '{},{},{:^4.1f},{:^4.1f},{:^4.1f},{:^4.1f},{:^4.1f},{:^4.1f}\n'.format(
        model_category,
        test_category,
        model_accuracy[RunOption.ALL_WORDS][algo] * 100,
        model_accuracy[RunOption.EMOTION_WORDS][algo] * 100,
        model_accuracy[RunOption.EXCLUDE_EMOTION_WORDS][algo] * 100,
        model_accuracy[RunOption.REVIEW_LENGTH][algo] * 100,
        model_accuracy[RunOption.SENTENCE_LENGTH][algo] * 100,
        model_accuracy[RunOption.POS][algo] * 100)

    with open(file_name, mode='a+') as f:
        f.seek(0)
        lines = f.readlines()[1:]

        for i in range(len(lines)):
            if lines[i].startswith(model_category + ',' + test_category + ','):
                lines[i] = entry
                break
        else:
            lines.append(entry)
            lines.sort()

        f.seek(0)
        f.truncate()
        f.write('model category,test category,'
                'all words,emotions only,exclude emotions,'
                'review length,sentence length,part of speech\n')
        f.writelines(lines)


def build_model(reviews, num_model_reviews, title, run_option):
    print('\n[' + title + '] Building models...')
    results = {}

    x_train, x_test, y_train, y_test = get_data_sets(reviews, num_model_reviews, run_option)

    if RUN_LOG_REG:
        print('[' + title + '] Running Logistic Regression...')
        results[Algo.LOG_REG] = logistic_regression(title, x_train, x_test, y_train, y_test)

    if RUN_NAIVE_BAYES:
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
parser.add_argument('path', type=str, help='reviews path')
parser.add_argument('test_path', type=str, nargs='?', default='', help='test reviews path (optional)')
args = parser.parse_args()
path = args.path
test_path = args.test_path
print('\nReviews path: ' + path)
print('Test reviews path: ' + test_path)

# load reviews
print('\nLoading reviews...')
reviews = []
load_reviews(path, reviews)
num_model_reviews = len(reviews)

# load test reviews
if test_path:
    print('\nLoading test reviews...')
    load_reviews(test_path, reviews)

# build models
results = {
    RunOption.ALL_WORDS: build_model(reviews, num_model_reviews, 'all words', RunOption.ALL_WORDS),
    RunOption.EMOTION_WORDS: build_model(reviews, num_model_reviews, 'emotion words only', RunOption.EMOTION_WORDS),
    RunOption.EXCLUDE_EMOTION_WORDS: build_model(reviews, num_model_reviews, 'exclude emotion words',
                                                 RunOption.EXCLUDE_EMOTION_WORDS),
    RunOption.REVIEW_LENGTH: build_model(reviews, num_model_reviews, 'review length', RunOption.REVIEW_LENGTH),
    RunOption.SENTENCE_LENGTH: build_model(reviews, num_model_reviews, 'sentence length', RunOption.SENTENCE_LENGTH),
    RunOption.POS: build_model(reviews, num_model_reviews, 'part of speech', RunOption.POS),
}

# calculate accuracy
model_accuracy = {
    RunOption.ALL_WORDS: calculate_accuracy(results[RunOption.ALL_WORDS]),
    RunOption.EMOTION_WORDS: calculate_accuracy(results[RunOption.EMOTION_WORDS]),
    RunOption.EXCLUDE_EMOTION_WORDS: calculate_accuracy(results[RunOption.EXCLUDE_EMOTION_WORDS]),
    RunOption.REVIEW_LENGTH: calculate_accuracy(results[RunOption.REVIEW_LENGTH]),
    RunOption.SENTENCE_LENGTH: calculate_accuracy(results[RunOption.SENTENCE_LENGTH]),
    RunOption.POS: calculate_accuracy(results[RunOption.POS]),
}

# print results
if RUN_LOG_REG:
    print_results('Logistic Regression', path, test_path, model_accuracy, Algo.LOG_REG)
if RUN_NAIVE_BAYES:
    print_results('Naive Bayes', path, test_path, model_accuracy, Algo.NAIVE_BAYES)

# save results
if os.path.isdir(path):
    model_category = 'All Categories'
else:
    start_idx = path.find('_')
    end_idx = path.find('_5')
    model_category = path[start_idx + 1:end_idx].replace('_', ' ')

test_category = model_category
if test_path:
    start_idx = path.find('_')
    end_idx = path.find('_5')
    test_category = test_path[start_idx + 1:end_idx].replace('_', ' ')

if RUN_LOG_REG:
    save_results('log_reg.csv', model_category, test_category, model_accuracy, Algo.LOG_REG)
if RUN_NAIVE_BAYES:
    save_results('naive_bayes.csv', model_category, test_category, model_accuracy, Algo.NAIVE_BAYES)

# elapsed time
print('\n\nElapsed time:', round(time.perf_counter() - start), 's')
