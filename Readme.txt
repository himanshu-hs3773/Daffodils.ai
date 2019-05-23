Machine Learning Project -
Nipun Dixit nd1462- Himanshu Singh hs3773



Text Generator Model with LSTM Recurrent Neural Network


In this Project firstly we try to develop a Word-Level Neural Network model and used it to Generate the text where it will predict the probability of the occurence of the next word in the sequence which will be based upon the word which has already been observed in the sequence.

Thereafter we will push our understanding to further develop a text generator with LSTM Recurrent Neural Network with the help of Keras, as Recurrent Neural Networks are generally used for the generative models but other than that they can be used as a predictive model which can learn the sequence of the problem and hence entirely generate new sequences.
Generative models are used to study the effectiveness of the model which has learned the problem and also are helpful in learning more about the problem domain.
In this project we will further try to create a generative model for the text, which will be done character by character by using LSTM recurrent neural network with Keras

Developing a Word-Level Neural Language Model and using it to Genrate Text
Neural networks are generally preferred in the development of the statistical language models as they can utilise a distributed representation where different words with similar meanings have similar representation and apart from that they can also use a larger context of recently observed words in making the predictions.



First Part- Developing a Word-Level Neural Language Model and using it to Genrate Text

Generated Tokens

['the', 'strange', 'case', 'of', 'dr', 'jekyll', 'and', 'mr', 'hyde', 'by', 'robert', 'louis', 'stevenson', 'contents', 'story', 'of', 'the', 'door', 'search', 'for', 'mr', 'hyde', 'dr', 'jekyll', 'was', 'quite', 'at', 'ease', 'the', 'carew', 'murder', 'case', 'incident', 'of', 'the', 'letter', 'incident', 'of', 'dr', 'lanyon', 'incident', 'at', 'the', 'window', 'the', 'last', 'night', 'dr', 'narrative', 'henry', 'full', 'statement', 'of', 'the', 'case', 'story', 'of', 'the', 'door', 'mr', 'utterson', 'the', 'lawyer', 'was', 'a', 'man', 'of', 'a', 'rugged', 'countenance', 'that', 'was', 'never', 'lighted', 'by', 'a', 'smile', 'cold', 'scanty', 'and', 'embarrassed', 'in', 'discourse', 'backward', 'in', 'sentiment', 'lean', 'long', 'dusty', 'dreary', 'and', 'yet', 'somehow', 'lovable', 'at', 'friendly', 'meetings', 'and', 'when', 'the']
Total Tokens: 24550
Unique Tokens: 3871

Parameter after fitting the model
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 50, 50)            193600    
_________________________________________________________________
lstm_3 (LSTM)                (None, 50, 100)           60400     
_________________________________________________________________
lstm_4 (LSTM)                (None, 100)               80400     
_________________________________________________________________
dense_3 (Dense)              (None, 100)               10100     
_________________________________________________________________
dense_4 (Dense)              (None, 3872)              391072    
=================================================================
Total params: 735,572
Trainable params: 735,572
Non-trainable params: 0
_________________________________________________________________

Some epochs value-
Epoch 1/100
24499/24499 [==============================] - 41s 2ms/step - loss: 6.6255 - acc: 0.0625
Epoch 2/100
24499/24499 [==============================] - 37s 2ms/step - loss: 6.1855 - acc: 0.0652
Epoch 3/100
24499/24499 [==============================] - 37s 2ms/step - loss: 6.0151 - acc: 0.0680
Epoch 4/100
24499/24499 [==============================] - 37s 2ms/step - loss: 5.8692 - acc: 0.0827
Epoch 5/100
24499/24499 [==============================] - 37s 1ms/step - loss: 5.7742 - acc: 0.0868
Epoch 6/100
24499/24499 [==============================] - 37s 1ms/step - loss: 5.7004 - acc: 0.0894
Epoch 7/100
24499/24499 [==============================] - 37s 1ms/step - loss: 5.6356 - acc: 0.0933
Epoch 8/100
24499/24499 [==============================] - 37s 1ms/step - loss: 5.5742 - acc: 0.0963
Epoch 9/100
24499/24499 [==============================] - 37s 1ms/step - loss: 5.5171 - acc: 0.0980
Epoch 10/100
24499/24499 [==============================] - 37s 1ms/step - loss: 5.4635 - acc: 0.1002

Final generated output of our model which is new and unique text

The first output paragraph is the printed seed text
Then the 50 words of the generated text are printed.

it almost rivalled the brightness of hope i was stepping leisurely across the court after breakfast drinking the chill of the air with pleasure when i was seized again with those indescribable sensations that heralded the change and i had but the time to gain the shelter of my cabinet before

two ahead now anatomical was long overthrown the screaming and obligation at least fumes of smiling and to be forced and you are not see it was a fine clear to the lawyer have been learning his clasped room i had come up with a tempest and the whole business



Second Part of the Project- Text Generator with LSTM Recurrent Neural Network with Keras-

1) Implementing Smaller Neural Network

Total Characters:  1140732
Total Vocab:  28
Total Patterns:  1140632

2) Implementing Larger Neural Network

We had got the results in the above sections but those were not so accurate results so now we will try to improve the performance or quality of the generated text by creating a much larger network.
We try to keep number of memory units to 256 but we will add second layer to our network



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
