from sklearn.feature_extraction.text import TfidfVectorizer
from DataReader import DataReader
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD

data_reader = DataReader("/Users/vedant/Desktop/emoji-detection/data/us_trial.text", "/Users/vedant/Desktop/emoji-detection/data/us_trial.labels")

tweets = data_reader.read_tweets()
labels = data_reader.read_labels()

for i in range(0,len(tweets)):
    tweet = tweets[i]
    stops = set(stopwords.words("english"))
    words = tweet.split(" ")
    str = ""
    for word in words:
        if word[0] != "@" and word not in stops:
            str += word+" "
    tweets[i] = str




tfidf = TfidfVectorizer()
print(tfidf)
x = tfidf.fit_transform(tweets)
print(x.shape[0])
print(x.shape[1])
svd = TruncatedSVD(n_components=300)
ret_val = svd.fit_transform(x)
print(ret_val.shape[0])
print(ret_val.shape[1])
print(len(ret_val[0]))

