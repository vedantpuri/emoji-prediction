# README #
### About
This is the repository for our NLP(CS-585) semester project at UMass Amherst. Our task is to suggest emojis based on a given tweet. This project is pertains to the top 20 emojis provided to us by SemEval 2018. The dataset that we used was also provided to us by SemEval. We apply several models in our task which have been briefly described below.
 


### Code division

#### Preprocessing etc. Code 
* `file_paths.py`
* `DataReader.py`


#### Models' Code 
* `NaiveBayes.py`
* `decision_tree_classifier.py`
* `LSTM.py`
* `bidirectional_lstm.py`

### Brief description of each file
#### file_paths.py
This file basically contains all the file paths that we need to use in our testing and training, and hence we create global variables for the file paths and use them across the various other files.

#### DataReader.py
Contains the main class which helps out in the preprocessing of text. Contains a few static and non static functions which are useful in many other models in our project. Also shuffles the data so as to randomize our training.

#### Naive Bayes.py
This was our baseline model. The code in this is very similar to what we did in HW1, except that this is a multinmoial Naive Baye's classifier.We simply use a bag of words representation in this and achieve an accuracy of about 26%

#### decision_tree_classifier.py
While making progress we decided to try out the decision tree classifier and use scikit learn to implement this for our dataset. We modified the format of data to pass it to the model and were able to touch an accuracy of 23.3%. Due to computational limitation we could only train on about 4000 tweets and test on a 1000.

#### LSTM.py
This is our implementation of an LSTM using Keras's neural network library with Tensorflow as the backend. The model was able to achieve 32.24% accuracy in a single epoch.
 

#### bidirectional_lstm.py
This is our implementation of a BLSTM using Keras's neural network library with Tensorflow as the backend. The model was able to achieve 33.37% accuracy in a single epoch.
 


### How to get the code running ###
#### Naivebayes.py
* **Initialization** - Create an object of NaiveBayes. The constructor takes in 3 parameters: the tweet file, the label file, test set ratio(0.2 by default).
* **Training** - Call the update model function which trains on the files provided above.
* **Testing** - Call the `evaluate_classifier_accuracy()` function which tests on the testing ratio and returns an accuracy.

**NOTE** - This has already been written out once at the bottom of the file, so you could just run the file as is from an IDE or from the terminal without any arguments passed in and it would work fine. It imports `file_paths.py`, `DataReader.py`, `math`, `defaultdict`and would pringt out the accuracy once its done.

#### decision_tree_classifier.py
This is much simpler to run as compared to Naive Bayes. A simple call to `run_dtc(twitter_file_path, label_file_path)` does the job. Within it it calls the other of the methods.
##### Inside `run_dtc()`
* **Initialization** - Inside our `run_dtc()` function we make a call to `populate_features()` which gives us the features and labels as np arrays which is the required format for sklearn decision tree classifier. We then split the features into training set and testing set. `split_train()` takes as parameters the original feature numpy arrays and a third parameter specifying the number of tweets to train on.
* **Training** - We use the sklearn classifier and hence we just pass on our training numpy arrays to the fit function.
* **Testing** - We now form our testing set of numpy arrays for each the tweets and labels by calling our `split_test()` function which takes in the original features X and Y, the starting number of testing, the ending number of testing range. We finally call our `evaluate_accuracy()` function which takes in our classifier, testing numpy arrays and the number of items we are testing on.

**NOTE** - This has already been written out once at the bottom of the file, so you could just run the file as is from an IDE or from the terminal without any arguments passed in and it would work fine. It imports `file_paths.py`, `DataReader.py`, `numpy`, `tree` from `sklearn` and would pringt out the accuracy once its done.


#### LSTM.py
* **All In 1 Go** - Simply call `run_LSTM(tweet_file_path, label_file_path)`. The function returns the accuracy for that run. The run will be for 1 epoch, during which the model will train on 40,000 tweets and test on 10,000 tweets.
* Example: accuracy = `run_LSTM(file_paths.us_tweets_path, file_paths.us_labels_path)`
*  		   print(accuracy)

#### bidirectional_lstm.py
* **Same as LSTM.py, All In 1 Go** - Simply call `run_LSTM(tweet_file_path, label_file_path)`. The function returns the accuracy for that run. The run will be for 1 epoch, during which the model will train on 40,000 tweets and test on 10,000 tweets.
* Example: accuracy = `run_LSTM(file_paths.us_tweets_path, file_paths.us_labels_path)`
*  		   print(accuracy)

### Contributors & Contact
* **Vedant Puri** - vedantpuri@umass.edu
* **Shubham Mehta** - shubhammehta@umass.edu
* **Ronit Arora** - rarora@umass.edu
