from collections import defaultdict
from nltk.corpus import stopwords
import random
import numpy as np

class DataReader:
    def __init__(self, tweet_file, labels_file):
        self.tweet_file = tweet_file
        self.labels_file = labels_file
        self.label_set = set()
        self.seed = random.randint(1, 10000001)

    def read_tweets(self):
        with open(self.tweet_file, 'r') as doc:
            tweets = doc.read().splitlines()
        random.seed(self.seed)
        random.shuffle(tweets)
        return tweets

    def read_labels(self):
        with open(self.labels_file, 'r') as doc:
            labels_for_tweets = doc.read().splitlines()
        random.seed(self.seed)
        random.shuffle(labels_for_tweets)
        self.label_set = set(labels_for_tweets)
        return labels_for_tweets

    def get_label_set(self):
        return self.label_set

    @staticmethod
    def tokenize(tweet):
        stops = set(stopwords.words("english"))
        bag_of_words = defaultdict(float)
        words = DataReader.extract_words_from_tweet(tweet)
        for word in words:
            if word[0] != "#" and word[0] != "@" and word not in stops:
                bag_of_words[word] += 1.0
        return bag_of_words

    @staticmethod
    def extract_words_from_tweet(tweet):
        word_list = []
        word_string = ""
        for char in tweet:
            if char.isalpha():
                word_string = word_string + char
            elif not len(word_string) < 2:
                word_list.append(word_string.lower())
                word_string = ""
        return word_list

    def get_features(self):
        tweets = self.read_tweets()
        labels = self.read_labels()
        np_labels = np.array(labels)
        vocab = set()
        for tweet in tweets:
            tokens = DataReader.tokenize(tweet).keys()
            for key in tokens:
                vocab.add(key)
        index_of_word = 0
        mapping = {}
        for word in vocab:
            mapping[word] = index_of_word
            index_of_word += 1

        feature_vector = np.zeros((len(tweets), len(vocab)), dtype=np.uint8)
        counter = 0
        for tweet in tweets:
            tokens = DataReader.tokenize(tweet).keys()
            for key in tokens:

                feature_vector[counter][mapping[key]] += 1
            counter += 1
        return feature_vector, np_labels

    @staticmethod
    def get_tokens(bow):
        """
        Returns the number of tokens in the bag of words.
        bow - bag of words representation
        """
        sum_toks = 0.0
        for key in bow:
            sum_toks += bow[key]
        return sum_toks


