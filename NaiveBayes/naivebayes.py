"""Volume 3: Naive Bayes Classifiers."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from collections import defaultdict
from collections import Counter
from sklearn.pipeline import make_pipeline


class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages into spam or ham.
    '''
    # Problem 1
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and P(x_i|C) to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        self.ham_probs = {}
        self.spam_probs = {}
        training_size = X.shape[0]
        words = set(X.str.split().explode())
        
        # helper function :)
        def compute_probs(class_filter):
            class_data = X[class_filter]
            exploded_data = class_data.str.split().explode()
            word_counts = exploded_data.value_counts()
            total_count = exploded_data.shape[0]
            return {word: (word_counts.get(word, 0) + 1) / (total_count + 2) for word in words}
        
        # compute probs
        self.prob_ham = y[y == "ham"].shape[0] / training_size
        self.ham_probs = compute_probs(y == "ham")
        
        self.prob_spam = y[y == "spam"].shape[0] / training_size
        self.spam_probs = compute_probs(y == "spam")
        
        return self


    # Problem 2
    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # initialize
        log_prob_spam = np.log(self.prob_spam)
        log_prob_ham = np.log(self.prob_ham)

        # helper function :) :)
        def calc_prob(row, prob_dict, base_prob):
            return sum(np.log(prob_dict.get(word, 1/2)) for word in row.split()) + base_prob

        # calculate final probability
        final_probs = [[calc_prob(row, self.ham_probs, log_prob_ham),
                        calc_prob(row, self.spam_probs, log_prob_spam)] for row in X]

        return np.array(final_probs)

    # Problem 3
    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # get probabilities & predictions
        probabilities = self.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        labels = np.array(["ham" if pred == 0 else "spam" for pred in predictions])
        
        return labels

def prob4():
    """
    Create a train-test split and use it to train a NaiveBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    # load in data
    df = pd.read_csv('sms_spam_collection.csv')
    X, y = df.Message, df.Label

    # split data & build model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb = NaiveBayesFilter()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    spam_correct = ((predictions == y_test) & (y_test == "spam")).sum()
    ham_wrong = ((predictions != y_test) & (y_test == "ham")).sum()

    spam_count = (y_test == "spam").sum()
    ham_count = (y_test == "ham").sum()

    return (spam_correct / spam_count, ham_wrong / ham_count)

# Problem 5
class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables.
    '''
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and r_{i,k} to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # initialize
        words = set(X.str.split().explode())
        training_size = X.shape[0]

        def compute_rates(class_filter):
            class_data = X[class_filter].str.split().explode()
            word_counts = class_data.value_counts()
            total_count = class_data.shape[0]
            return {word: (word_counts.get(word, 0) + 1) / (total_count + 2) for word in words}, total_count

        self.ham_rates, self.ham_amount = compute_rates(y == "ham")
        self.spam_rates, self.spam_amount = compute_rates(y == "spam")
        self.prob_ham = (y == "ham").sum() / training_size
        self.prob_spam = (y == "spam").sum() / training_size

        return self

    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # helper function
        def calc_prob(row, rate_dict, total_amount, base_prob):
            word_counts = Counter(row.split())
            n = len(row.split())
            return sum(np.log(stats.poisson.pmf(word_counts[word], rate_dict.get(word, 1 / (total_amount + 2)) * n)) for word in set(row.split())) + base_prob

        log_prob_ham = np.log(self.prob_ham)
        log_prob_spam = np.log(self.prob_spam)

        log_prob = np.array([[calc_prob(row, self.ham_rates, self.ham_amount, log_prob_ham),
                          calc_prob(row, self.spam_rates, self.spam_amount, log_prob_spam)] for row in X])

        return log_prob

    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # calc probs
        probabilities = self.predict_proba(X)
        labels = np.array(["ham" if pred == 0 else "spam" for pred in np.argmax(probabilities, axis=1)])
        return labels

def prob6():
    """
    Create a train-test split and use it to train a PoissonBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    # load in data
    df = pd.read_csv('sms_spam_collection.csv')
    X, y = df.Message, df.Label

    # split data & build model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pb = PoissonBayesFilter()
    pb.fit(X_train, y_train)
    predictions = pb.predict(X_test)

    # calculate
    spam_correct = ((predictions == y_test) & (y_test == "spam")).sum()
    ham_wrong = ((predictions != y_test) & (y_test == "ham")).sum()

    spam_count = (y_test == "spam").sum()
    ham_count = (y_test == "ham").sum()

    return (spam_correct / spam_count, ham_wrong / ham_count)
    
# Problem 7
def sklearn_naive_bayes(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    # create & train the pipeline
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)

    # predict
    predictions = model.predict(X_test)

    return np.array(predictions)

def test1():
    df = pd.read_csv('sms_spam_collection.csv')
    # separate the data
    X, y = df.Message, df.Label
    # build classifier
    nb = NaiveBayesFilter()
    nb.fit(X[:300], y[:300])
    print(nb.ham_probs['out'])
    print(nb.spam_probs['out'])

def test2():
    df = pd.read_csv('sms_spam_collection.csv')
    # separate the data
    X, y = df.Message, df.Label
    # build classifier
    nb = NaiveBayesFilter()
    nb.fit(X[:300], y[:300])
    print(nb.predict_proba(X[800:805]))

def test3():
    df = pd.read_csv('sms_spam_collection.csv')
    # separate the data
    X, y = df.Message, df.Label
    # build classifier
    nb = NaiveBayesFilter()
    nb.fit(X[:300], y[:300])
    print(nb.predict(X[800:805]))

def test5():
    df = pd.read_csv('sms_spam_collection.csv')
    # separate the data
    X, y = df.Message, df.Label
    # build classifier
    pb = PoissonBayesFilter()
    pb.fit(X[:300], y[:300])
    print(pb.ham_rates['in'])
    print(pb.spam_rates['in'])
    print(pb.predict_proba(X[800:805]))
    print(pb.predict(X[800:805]))

def test7():
    df = pd.read_csv('sms_spam_collection.csv')
    accuracy = {"spam_correct": 0, "ham_wrong":0}
    # separate the data
    X, y = df.Message, df.Label
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(sklearn_naive_bayes(X_train, y_train,X[800:805]))