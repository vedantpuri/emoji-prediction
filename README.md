![logo](resources/header.png)

![Status](https://img.shields.io/pypi/status/Django.svg?style=for-the-badge) ![GitHub repo size in bytes](https://img.shields.io/github/repo-size/vedantpuri/emoji-prediction.svg?style=for-the-badge)

## About
In today’s world where short messages and tweets are at the core of communication, emojis have become major forms of expression of ideas and emotions. They penetrate language barriers and allow people to express a whole lot in a very concise manner. With the increasing use of emojis in our daily life, sometimes we lose context of text and aren’t really sure about which emoji to use based on the text. Our project aims to suggest emojis based on the given text by analyzing the sentiment of the given text and predicting relevant emojis for it.

## Requirements
 - Python >= 3
 - NLTK >= 3.2.3
 - Keras >= 2.0.7
 - Word Embeddings
    - Download [here](https://drive.google.com/open?id=0B13VF_-CUsHPN0dveFZBODlUU00)
    - Place them in the src folder

## Methods Employed
 - Naive Bayes Classifier (`nb`)
 - Decision Tree Classifier (`dtc`)
 - LSTM (`lstm`)
 - Bi-Directional LSTM (`blstm`)

## Usage
For information on running the project and further knowledge of the implementation, strategies and accuracies received, please refer to the [UNDERSTANDING.md](https://github.com/vedantpuri/emoji-prediction/blob/master/UNDERSTANDING.md) file.

## License
The project is available under the **MIT** License. Check the [license ](https://github.com/vedantpuri/emoji-prediction/blob/master/LICENSE.md) file for more information.
