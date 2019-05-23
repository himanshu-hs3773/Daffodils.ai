Machine Learning Project -
Nipun Dixit nd1462- Himanshu Singh hs3773



Text Generator Model with LSTM Recurrent Neural Network


In this Project firstly we try to develop a Word-Level Neural Network model and used it to Generate the text where it will predict the probability of the occurence of the next word in the sequence which will be based upon the word which has already been observed in the sequence.

Thereafter we will push our understanding to further develop a text generator with LSTM Recurrent Neural Network with the help of Keras, as Recurrent Neural Networks are generally used for the generative models but other than that they can be used as a predictive model which can learn the sequence of the problem and hence entirely generate new sequences.
Generative models are used to study the effectiveness of the model which has learned the problem and also are helpful in learning more about the problem domain.

In this project we will further try to create a generative model for the text, which will be done character by character by using LSTM recurrent neural network with Keras
Developing a Word-Level Neural Language Model and using it to Genrate Text
Neural networks are generally preferred in the development of the statistical language models as they can utilise a distributed representation where different words with similar meanings have similar representation and apart from that they can also use a larger context of recently observed words in making the predictions.



Following are the further Ideas which can be used in the working of our model

--Predicting fewer than 1,000 characters as output for a given seed.
--Remove all punctuation from the source text, and therefore from the models’ vocabulary.
--Try a one hot encoded for the input sequences.
--Train the model on padded sentences rather than random sequences of characters.
--Add more memory units to the layers and/or more layers.
--Experiment with scale factors (temperature) when interpreting the prediction probabilities.
--Changing the LSTM layers to be “stateful” to maintain state across batches.
--By Increasing the number of training epochs to 100 or many hundreds.
--Adding the dropout to the visible input layer and considering tuning of the dropout percentage.
--Tuning the batch size, trying a batch size of 1 as a baseline and then choosing the larger sizes from there.


Resources and Referrences Used

This character text model is a very popular way for generating text using recurrent neural networks.
Below are some resources which were used in developing this project

--Generating Text with Recurrent Neural Networks [pdf], 2011 (http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)
--MXNet tutorial for using an LSTM for text generation. (http://mxnetjl.readthedocs.io/en/latest/tutorial/char-lstm.html)
--Keras code example of LSTM for text generation. (https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)
-- Text Genration in python with Keras (https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
--Lasagne code example of LSTM for text generation. (https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py)


Summary
In this project we could discover how we can efficiently develop and train a LSTM recurrent neural network model for text generation with the Keras which is a deep learning library.

These 3 goals and learnings were primarily have been achieved after implementing the first part of our project-
-- How to prepare text for developing a word-based language model.
-- How to design and fit a neural language model with a learned embedding and an LSTM hidden layer. -- How to use the learned language model to generate new text with similar statistical properties as the source text.

These 3 goals and learnings were primarily have been achieved after implementing the second part of our project:
--How to train an LSTM network on text sequences
--How to use the trained network to generate new sequences. --How to develop stacked LSTM networks and lift the performance of the model.
