import argparse
import gzip
import json
import string
import time

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB

# header length
HEADER_LEN = 80

# max reviews
MAX_REVIEWS = 10000

# punctuation table
punctuation_table = str.maketrans('', '', string.punctuation)

# Porter stemmer
stemmer = PorterStemmer()

# WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# stop words
stop_words = set(stopwords.words('english'))

# emotion words
EMOTION_WORDS_FILE_NAME = 'emotions.txt'


def load_emotion_words():
    with open(EMOTION_WORDS_FILE_NAME, mode='r') as f:
        return set(stemmer.stem(line.strip()) for line in f)


emotion_words = load_emotion_words()


def stem(tokens):
    return [stemmer.stem(token) for token in tokens]


def lemmatize(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


def remove_stopwords(tokens):
    return [token for token in tokens if token not in stop_words]


def extract_emotion_words(tokens):
    return [token for token in tokens if token in emotion_words]


def exclude_emotion_words(tokens):
    return [token for token in tokens if token not in emotion_words]


def cleanup(sentence):
    tokens = word_tokenize(sentence)

    # remove punctuation
    tokens = [w.translate(punctuation_table) for w in tokens]

    # remove non-alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]

    # lowercase
    tokens = [token.lower() for token in tokens]

    sentence = ' '.join(tokens)
    return sentence


def load_reviews(file_name):
    reviews = []

    with gzip.open(file_name, mode='r') as f:
        for i, line in enumerate(f):
            if i == MAX_REVIEWS:
                break
            print('Loading review #', i)
            entry = json.loads(line)
            review = entry['reviewText']
            helpful = entry['helpful']
            helpful_votes = helpful[0]
            total_votes = helpful[1]
            if helpful_votes * 2 > total_votes > 1:
                helpfulness = 'helpful'
            else:
                helpfulness = 'not_helpful'
            # print(str(helpful) + ' ' + helpfulness + ': ' + review)
            review = cleanup(review)
            reviews.append((review, helpfulness))

    return reviews


def preprocess(reviews, type, em, xem):
    preprocessed_reviews = []

    for i, (review, helpfulness) in enumerate(reviews):
        print('[' + type + '] Processing review #', i)
        tokens = word_tokenize(review)
        tokens = remove_stopwords(tokens)
        tokens = stem(tokens)  # tokens = lemmatize(tokens)
        if em:
            tokens = extract_emotion_words(tokens)
        elif xem:
            tokens = exclude_emotion_words(tokens)
        review = ' '.join(tokens)
        preprocessed_reviews.append((review, helpfulness))

    return preprocessed_reviews


def get_data_sets(reviews):
    data, labels = map(list, zip(*reviews))
    vocabulary_size = len(set(token for sentence in data for token in word_tokenize(sentence)))

    # extract features
    rank_threshold = 100
    max_features = vocabulary_size - rank_threshold if vocabulary_size > rank_threshold else None
    vectorizer = CountVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(data)

    # split reviews_helpfulness into train and test reviews_helpfulness sets
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        train_size=0.9,
        random_state=2797)
    return X_train, X_test, y_train, y_test


def logistic_regression(title, reviews):
    X_train, X_test, y_train, y_test = get_data_sets(reviews)

    # create hyperparameter options
    C = np.logspace(-4, 4, num=10)
    penalty = ['l1', 'l2']
    hyperparameters = {'C': C, 'penalty': penalty}

    # create grid search using 5-fold cross validation
    log_reg = LogisticRegression(solver='liblinear')
    grid_obj = GridSearchCV(log_reg, hyperparameters, scoring='accuracy', cv=5, n_jobs=-1)

    # fit grid search
    grid_obj.fit(X=X_train, y=y_train)

    # retrieve best classifier
    classifier = grid_obj.best_estimator_

    # accuracy score
    accuracy = grid_obj.best_score_
    print_accuracy(title, accuracy)

    return {
        'acc': accuracy * 100,  # percentage
        'clf': classifier,
        'X_test': X_test,
        'y_test': y_test,
    }


def naive_bayes(title, reviews):
    X_train, X_test, y_train, y_test = get_data_sets(reviews)

    # train the model using the training set
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # 5-fold cross validation
    scores = cross_val_score(classifier, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1)

    # accuracy score
    accuracy = np.mean(scores)
    print_accuracy(title, accuracy)

    return {
        'acc': accuracy * 100,  # percentage
        'clf': classifier,
        'X_test': X_test,
        'y_test': y_test,
    }


def print_accuracy(title, accuracy):
    print_title(title)
    print('Accuracy: {:.1f}%\n'.format(accuracy * 100))


def print_title(title):
    header = '=' * HEADER_LEN
    print(header)
    print(title)
    print(header)


def print_results(model_accuracy):
    h1 = '=' * HEADER_LEN
    h2 = '-' * HEADER_LEN

    print()
    print('{:^80}'.format(file_name))
    print(h1)
    print('{:^1}{:^78}{:^1}'.format('|', 'Accuracy %', '|'))
    print(h2)
    print(
        '{:^1}{:^24}{:^3}{:^24}{:^3}{:^24}{:^1}'.format('|', 'all words', '|', 'emotion words only', '|',
                                                        'exclude emotion words',
                                                        '|'))
    print(h2)
    print('{:^1}{:^12}{:^12}{:^3}{:^12}{:^12}{:^3}{:^12}{:^12}{:^1}'
          .format('|', 'Log Reg', 'Naive Bayes', '|', 'Log Reg', 'Naive Bayes', '|', 'Log Reg', 'Naive Bayes', '|'))
    print(h1)

    fmt = '{:^1}{:^12.1f}{:^12.1f}{:^3}{:^12.1f}{:^12.1f}{:^3}{:^12.1f}{:^12.1f}{:^}'
    print(fmt.format('|', model_accuracy['all']['lr'] * 100, model_accuracy['all']['nb'] * 100, '|',
                     model_accuracy['em']['lr'] * 100, model_accuracy['em']['nb'] * 100, '|',
                     model_accuracy['xem']['lr'] * 100, model_accuracy['xem']['nb'] * 100, '|'))
    print(h2)


def build_and_train_model(original_reviews, type, em, xem):
    # preprocess reviews_helpfulness
    print('\n[' + type + '] Data preparation...')
    reviews = preprocess(original_reviews, type, em, xem)

    # build and train model
    print('\n[' + type + '] Building models...')
    results = {
        'lr': {'acc': 0, 'clf': None, 'X_test': None, 'y_test': None},
        'nb': {'acc': 0, 'clf': None, 'X_test': None, 'y_test': None},
    }

    title = 'Logistic Regression'
    print('\n[' + type + '] Running ' + title + '...\n')
    results['lr'] = logistic_regression(title, reviews)

    title = 'Naive Bayes'
    print('\n[' + type + '] Running ' + title + '...\n')
    results['nb'] = naive_bayes(title, reviews)

    return results


def calculate_accuracy(results):
    accuracy = {}

    for k, d in results.items():
        # retrieve the best classifier
        classifier = d['clf']
        X_test = d['X_test']
        y_test = d['y_test']

        # predict the response for the test set
        y_predict = classifier.predict(X_test)

        # accuracy score
        accuracy[k] = metrics.accuracy_score(y_test, y_predict)

    return accuracy


# start time
start_time = time.perf_counter()

# parameters
parser = argparse.ArgumentParser()
parser.add_argument('file_name', type=str, help='reviews_helpfulness file name')
args = parser.parse_args()
file_name = args.file_name

print_title('Parameters')
print('Data file name: ' + file_name)

# load reviews_helpfulness (helpful, not_helpful)
print('\nLoading reviews_helpfulness...')
original_reviews = load_reviews(file_name)

results = {}
results['all'] = build_and_train_model(original_reviews, 'all words', False, False)
results['em'] = build_and_train_model(original_reviews, 'emotion words only', True, False)
results['xem'] = build_and_train_model(original_reviews, 'exclude emotion words', False, True)

model_accuracy = {}
model_accuracy['all'] = calculate_accuracy(results['all'])
model_accuracy['em'] = calculate_accuracy(results['em'])
model_accuracy['xem'] = calculate_accuracy(results['xem'])

print_results(model_accuracy)

# elapsed time
print('\nElapsed time:', round(time.perf_counter() - start_time), 's')
