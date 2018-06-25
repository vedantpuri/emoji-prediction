from collections import defaultdict
import math
from DataReader import DataReader


class NaiveBayes:
    def __init__(self, input_tweet_file, input_labels_file, alpha=1, test_set_ratio=0.2):
        self.test_set_ratio = test_set_ratio
        self.alpha = alpha
        # Vocabulary is a set that stores every word seen in the training data
        self.vocabulary = set()

        # total_tweets_per_class is a dictionary that maps a class (i.e., 0/1/2/3 etc.) to
        # the number of tweets in the training set of that class
        self.total_tweets_per_class = defaultdict(float)

        # word_counts_per_class is a dictionary that maps a class (i.e., 0/1/2/3 etc.) to
        # the number of words in the training set in tweets of that class
        self.word_counts_per_class = defaultdict(float)

        # words_per_class is a dictionary of dictionaries. It maps a class (i.e.,
        # 0/1/2/3 etc.) to a dictionary of word counts. For example:
        #   self.words_per_class[0]['awesome']
        # stores the number of times the word 'awesome' appears in tweets
        # of the 0th emoji class in the training tweets.
        self.words_per_class = {}
        data = DataReader(input_tweet_file, input_labels_file)
        # data = DataReader("../data/us_trial.text", "../data/us_trial.labels")
        self.tweets = data.read_tweets()
        self.labels = data.read_labels()
        self.label_set = data.get_label_set()

        for label in self.label_set:
            self.words_per_class[label] = defaultdict(float)

        self.prior_count_tweets = 0.0

    def update_model(self):
        train_tweets = int(len(self.tweets) * (1 - self.test_set_ratio))
        self.prior_count_tweets = train_tweets
        for tweet_number in range(0, train_tweets):
            label = self.labels[tweet_number]
            self.total_tweets_per_class[label] += 1.0
            bow = DataReader.tokenize(self.tweets[tweet_number])
            sum = DataReader.get_tokens(bow)
            self.word_counts_per_class[label] += sum
            for key in bow:
                self.words_per_class[label][key] += bow[key]
                self.vocabulary.add(key)

    def p_word_given_label_and_pseudocount(self, word, label):
        """
        Returns the probability of word given label wrt psuedo counts.
        alpha - pseudocount parameter
        """
        den = self.alpha * len(self.vocabulary)
        my_word_prob = self.words_per_class[label][word] + self.alpha
        total_words_label = self.word_counts_per_class[label] + den
        return my_word_prob / total_words_label

    def log_likelihood(self, bow, label):
        """
        Computes the log likelihood of a set of words give a label and pseudocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; pseudocount parameter
        """
        log_lk = 0
        for key in bow.keys():
            log_lk += math.log(self.p_word_given_label_and_pseudocount(key, label))
        return log_lk

    def log_prior(self, label):
        """
        Returns the log prior of a document having the class 'label'.
        """
        c = self.total_tweets_per_class[label]
        tot = self.prior_count_tweets
        return math.log(c / tot)

    def unnormalized_log_posterior(self, bow, label):
        """
        Computes the unnormalized log posterior (of doc being of class 'label').
        bow - a bag of words (i.e., a tokenized document)
        """
        return self.log_prior(label) + self.log_likelihood(bow, label)

    def classify(self, bow):
        """
        Classifies a tweet based on it's bag of word representation
        bow - a bag of words (i.e., a tokenized document)
        alpha - pseudocount
        """
        max_unnormalized = float('-inf')
        argmax_unnormalized = '-1'
        for label in self.label_set:
            var_ret = self.unnormalized_log_posterior(bow, label)
            if var_ret > max_unnormalized:
                max_unnormalized = var_ret
                argmax_unnormalized = label
        return argmax_unnormalized

    # Running classifier on the test set.
    def evaluate_classifier_accuracy(self):
        correct = 0.0
        total = 0.0
        l = len(self.tweets)
        test_tweets = int(l * (1 - self.test_set_ratio))
        for tweet_num in range(test_tweets, l):
            label = self.labels[tweet_num]
            tweet = self.tweets[tweet_num]
            bow = DataReader.tokenize(tweet)
            predicted_label = self.classify(bow)
            if predicted_label == label:
                correct += 1.0
            total += 1.0

        return 100 * (correct / total)
