
## Code division

### Preprocessing etc. Code
* file_paths.py
* DataReader.py

### Models' Code
* NaiveBayes.py
* decision_tree_classifier.py
* LSTM.py
* bidirectional_lstm.py

## Brief description of each file
- `file_paths.py`  
This file basically contains all the file paths that we need to use in our testing and training, and hence we create global variables for the file paths and use them across the various other files.

- `DataReader.py`  
Contains the main class which helps out in the preprocessing of text. Contains a few static and non static functions which are useful in many other models in our project. Also shuffles the data so as to randomize our training.

- `NaiveBayes.py`  
This was our baseline model. The code in this is  is a multinmoial Naive Baye's classifier.We simply use a bag of words representation in this and achieve an accuracy of about 26%

- `decision_tree_classifier.py`  
While making progress we decided to try out the decision tree classifier and use scikit learn to implement this for our dataset. We modified the format of data to pass it to the model and were able to touch an accuracy of 23.3%. Due to computational limitation we could only train on about 4000 tweets and test on a 1000.

- `LSTM.py`  
This is our implementation of an LSTM using Keras's neural network library with Tensorflow as the backend. The model was able to achieve 32.24% accuracy in a single epoch.


- `bidirectional_lstm.py`  
This is our implementation of a BLSTM using Keras's neural network library with Tensorflow as the backend. The model was able to achieve 33.37% accuracy in a single epoch.

- `deploy_model,py`  
This is the main driver script to run any model of your choice. Simply calls required functions from other modules.
