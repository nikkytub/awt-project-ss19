# Advanced Web Technology Project SS-19 (AI based context and sentiment analysis for messaging)

It is a project based on Recurrent neural network using TensorFlow which is used to detect cyberbullying in the messages. 

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

Please make sure that you have installed Django version "2.2.2"


### Installing

Clone this repository in your local machine via
```
$ git clone git@gitlab.tubit.tu-berlin.de:nikhil.singh/awt-project-ss19.git
```

## Deployment

Please create virtual env for python, activate it and install the required 
dependencies. To train, test and save rnn model with tensorflow vocab processor 
embedding. Please run
```
$ python awt.py
```
To train, test and save RNN model with GloVe embedding. Please run
```
$ python glove.py
```
To train, test and save rnn model with Twitter dataset and tensorflow vocab processor 
embedding. Please run
```
$ python model_twitter.py
```
To train, test and save rnn model with Twitter dataset and GloVe embedding. 
Please run
```
$ python model_twitter_glove.py
```

To run django channels based chat app. Please install virtual env for python3 
with required dependencies. After that please run
```
$ python3 manage.py runserver
```
It will start the chat app on port 8000 in your local machine and use the saved 
RNN model. Simply go to "http://localhost:8000/chat" Now type the name of the 
chat room you would like to enter and start typing chat messages. Since the 
dataset used from MySpace and Twitter to train the model was quite small you may see 
unexpected behavior from the model. To check how well model is behaving please 
look at the test-accuracy.

## Dataset used
* MySpace dataset for cyberbullying
* Twitter dataset for cyberbullying

## Built With
* TensorFlow
* [Django](https://www.djangoproject.com/) - The web framework used


## Authors
* **Nikhil Singh**
* **Johanna Wallner**


## Acknowledgments

* Thanks to André Paul for his supervision