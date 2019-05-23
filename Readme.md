# Dafodils.ai- Text and Story Generator Tool #
Nipun Dixit nd1462- Himanshu Singh hs3773



# Text Generator Model with LSTM Recurrent Neural Network #

A text generator tool which generates new and unique text based upon a particular random theme (random seed) which is chosen and fed to the model to generate a text based upon that particular theme/seed.


In this Project firstly we tried to develop a Word-Level Neural Network model and used it to Generate the text where it will predict the probability of the occurence of the next word in the sequence which will be based upon the word which has already been observed in the sequence.

Thereafter we pushed our understanding to further develop a text generator with LSTM Recurrent Neural Network with the help of Keras and which is based upon a random seed which is chosen from the data and hence produce the new and unique text

As Recurrent Neural Networks are generally used for the generative models but other than that they can be used as a predictive model which can learn the sequence of the problem and hence entirely generate new sequences.
Generative models are used to study the effectiveness of the model which has learned the problem and also are helpful in learning more about the problem domain.

In this project we  further tried to create a generative model for the text, which will be done character by character by using LSTM recurrent neural network with Keras


## Developing a Word-Level Neural Language Model and using it to Genrate Text ##
Neural networks are generally preferred in the development of the statistical language models as they can utilise a distributed representation where different words with similar meanings have similar representation and apart from that they can also use a larger context of recently observed words in making the predictions.



### First Part- Developing a Word-Level Neural Language Model and using it to Genrate Text ###

#### Generated Tokens ####

['the', 'strange', 'case', 'of', 'dr', 'jekyll', 'and', 'mr', 'hyde', 'by', 'robert', 'louis', 'stevenson', 'contents', 'story', 'of', 'the', 'door', 'search', 'for', 'mr', 'hyde', 'dr', 'jekyll', 'was', 'quite', 'at', 'ease', 'the', 'carew', 'murder', 'case', 'incident', 'of', 'the', 'letter', 'incident', 'of', 'dr', 'lanyon', 'incident', 'at', 'the', 'window', 'the', 'last', 'night', 'dr', 'narrative', 'henry', 'full', 'statement', 'of', 'the', 'case', 'story', 'of', 'the', 'door', 'mr', 'utterson', 'the', 'lawyer', 'was', 'a', 'man', 'of', 'a', 'rugged', 'countenance', 'that', 'was', 'never', 'lighted', 'by', 'a', 'smile', 'cold', 'scanty', 'and', 'embarrassed', 'in', 'discourse', 'backward', 'in', 'sentiment', 'lean', 'long', 'dusty', 'dreary', 'and', 'yet', 'somehow', 'lovable', 'at', 'friendly', 'meetings', 'and', 'when', 'the']

Total Tokens: 24550

Unique Tokens: 3871

Total Sequences: 24499

#### Parameter after fitting the model ####
_________________________________________________________________
Layer (type)              |   Output Shape          |    Param    
--------------------------|-------------------------|------------
embedding_2 (Embedding)   |   (None, 50, 50)        |    193600    
                          |                         |
lstm_3 (LSTM)             |   (None, 50, 100)       |    60400     
                          |                         |
lstm_4 (LSTM)             |   (None, 100)           |    80400     
                          |                         |
dense_3 (Dense)           |   (None, 100)           |    10100     
                          |                         |
dense_4 (Dense)           |   (None, 3872)          |    391072    
-----------------------------------------------------------------
-----------------------------------------------------------------
Total params: 735,572
Trainable params: 735,572
Non-trainable params: 0
-----------------------------------------------------------------

#### Some epochs value ####
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

#### Final generated output of our model which is new and unique text ####

##### The first output paragraph is the printed seed text #####
##### Then the 50 words of the generated text are printed #####

#### Seed: ####
it almost rivalled the brightness of hope i was stepping leisurely across the court after breakfast drinking the chill of the air with pleasure when i was seized again with those indescribable sensations that heralded the change and i had but the time to gain the shelter of my cabinet before

#### Generated Text ####
two ahead now anatomical was long overthrown the screaming and obligation at least fumes of smiling and to be forced and you are not see it was a fine clear to the lawyer have been learning his clasped room i had come up with a tempest and the whole business



### Second Part of the Project- Text Generator with LSTM Recurrent Neural Network with Keras ###

In this part of our project we will focuss primarily on LSTM recurrent neural network to improve the performance and the quality of the text generation, here we will use another text dataset to perform the text generation operation.

So firstly Small LSTM Recurrent Neural Network is designed then later a Large LSTM Recurrent Neural Network is designed which will significantly improve the performance of the model

### 1) Implementing Smaller Neural Network ###

#### Parameter after fitting the model ####

Total Characters:  424971
Total Vocab:  49
Total Patterns:  424871

* In this step we defined our LSTM model where a single hidden LSTM layer is defined with 256 memory units, also the output layer was made dense by usage of softmax activation function such that output probability prediction is made of the 47 characters between 0 and 1

#### Some Epoch Values ####
Epoch 1/10
424871/424871 [==============================] - 629s 1ms/step - loss: 2.8798

Epoch 00001: loss improved from inf to 2.87977, saving model to weights-improvement-01-2.8798.hdf5
Epoch 2/10
424871/424871 [==============================] - 628s 1ms/step - loss: 2.6902

Epoch 00002: loss improved from 2.87977 to 2.69022, saving model to weights-improvement-02-2.6902.hdf5
Epoch 3/10
424871/424871 [==============================] - 631s 1ms/step - loss: 2.5705

Epoch 00003: loss improved from 2.69022 to 2.57052, saving model to weights-improvement-03-2.5705.hdf5
Epoch 4/10
424871/424871 [==============================] - 626s 1ms/step - loss: 2.4808

Epoch 00004: loss improved from 2.57052 to 2.48084, saving model to weights-improvement-04-2.4808.hdf5
Epoch 5/10
424871/424871 [==============================] - 628s 1ms/step - loss: 2.4111

Epoch 00005: loss improved from 2.48084 to 2.41106, saving model to weights-improvement-05-2.4111.hdf5
Epoch 6/10
424871/424871 [==============================] - 626s 1ms/step - loss: 2.3534

Epoch 00006: loss improved from 2.41106 to 2.35340, saving model to weights-improvement-06-2.3534.hdf5
Epoch 7/10
424871/424871 [==============================] - 628s 1ms/step - loss: 2.3018

Epoch 00007: loss improved from 2.35340 to 2.30184, saving model to weights-improvement-07-2.3018.hdf5
Epoch 8/10
424871/424871 [==============================] - 629s 1ms/step - loss: 2.2579

Epoch 00008: loss improved from 2.30184 to 2.25792, saving model to weights-improvement-08-2.2579.hdf5
Epoch 9/10
424871/424871 [==============================] - 619s 1ms/step - loss: 2.2191

Epoch 00009: loss improved from 2.25792 to 2.21908, saving model to weights-improvement-09-2.2191.hdf5
Epoch 10/10
424871/424871 [==============================] - 612s 1ms/step - loss: 2.1817
Epoch 00010: loss improved from 2.21908 to 2.18173, saving model to weights-improvement-10-2.1817.hdf5




#### Final generated output of our model which is new and unique text ####

##### The first output paragraph is the printed seed text #####
##### Then the 50 words of the generated text are printed. #####

#### Seed: ####
" couldn't without passports and things. besides i've seen that man,
boris something, since. he dined  "

#### Generated Text Based upon particular theme/seed ####

to the toaee th the tay an a fote to the poaee to the poaee to the soaee of the gorse if the soaee of the poaee th the taale th the taale that she was a soacl oo the soaee of the gorse th the tas ao a fert of the poaee thet she had neter teeled to the soaee of the soaee of the poaee th the taale th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poaee th the taale thet the was a boeek of the poae
Done.

#### Interpreting the result obtained ####

* It conforms to the format of the line observed in the original text which is less than total 80 characters before the generation of the new line.

* The word-like groups are made by the separation of the characters where most are seeming to be English words (e.g. “the”,     “little” and “was”), but many even do not seem to be english word (e.g. “boeek”, “taale” and “thet”).

* Upon the observation it looks like some of the words obtained in the sequence make sense (e.g. “of the poae“), but many do not make any sense (e.g. “taale thet the“).



### 2) Implementing Larger Neural Network ###

We had got the results in the above sections but those were not so accurate results so now we will try to improve the performance or quality of the generated text by creating a much larger network.
We try to keep number of memory units to 256 but we will add second layer to our network

#### Parameter after fitting the model ####  

Total Characters:  424971
Total Vocab:  49
Total Patterns:  424871
#### Some Epoch Values ####
Epoch 1/10
424871/424871 [==============================] - 626s 1ms/step - loss: 2.8109

Epoch 00001: loss improved from inf to 2.81087, saving model to weights-improvement-01-2.8109-bigger.hdf5
Epoch 2/10
424871/424871 [==============================] - 618s 1ms/step - loss: 2.4743

Epoch 00002: loss improved from 2.81087 to 2.47430, saving model to weights-improvement-02-2.4743-bigger.hdf5
Epoch 3/10
424871/424871 [==============================] - 631s 1ms/step - loss: 2.2677

Epoch 00003: loss improved from 2.47430 to 2.26767, saving model to weights-improvement-03-2.2677-bigger.hdf5
Epoch 4/10
424871/424871 [==============================] - 639s 2ms/step - loss: 2.1377

Epoch 00004: loss improved from 2.26767 to 2.13771, saving model to weights-improvement-04-2.1377-bigger.hdf5
Epoch 5/10
424871/424871 [==============================] - 637s 1ms/step - loss: 2.0516

Epoch 00005: loss improved from 2.13771 to 2.05160, saving model to weights-improvement-05-2.0516-bigger.hdf5
Epoch 6/10
424871/424871 [==============================] - 638s 2ms/step - loss: 1.9884

Epoch 00006: loss improved from 2.05160 to 1.98843, saving model to weights-improvement-06-1.9884-bigger.hdf5
Epoch 7/10
424871/424871 [==============================] - 636s 1ms/step - loss: 1.9372

Epoch 00007: loss improved from 1.98843 to 1.93721, saving model to weights-improvement-07-1.9372-bigger.hdf5
Epoch 8/10
424871/424871 [==============================] - 638s 2ms/step - loss: 1.8939

Epoch 00008: loss improved from 1.93721 to 1.89392, saving model to weights-improvement-08-1.8939-bigger.hdf5
Epoch 9/10
424871/424871 [==============================] - 643s 2ms/step - loss: 1.8582

Epoch 00009: loss improved from 1.89392 to 1.85817, saving model to weights-improvement-09-1.8582-bigger.hdf5
Epoch 10/10
424871/424871 [==============================] - 647s 2ms/step - loss: 1.8246
Epoch 00010: loss improved from 1.85817 to 1.82462, saving model to weights-improvement-10-1.8246-bigger.hdf5


#### Final generated output of our model which is new and unique text ####

##### The first output paragraph is the printed seed text #####
##### Then the 50 words of the generated text are printed. #####

##### Seed: #####
 ' ' d herself lying on the bank, with her head in the lap of her sister, who was gently brushing away s ''

##### Text Generator #####
 ' '
herself lying on the bank, with her head in the lap of her sister, who was gently brushing away so siee, and she sabbit said to herself and the sabbit said to herself and the sood way of the was a little that she was a little lad good to the garden, and the sood of the mock turtle said to herself, 'it was a little that the mock turtle said to see it said to sea it said to sea it say it the marge hard sat hn a little that she was so sereated to herself, and she sabbit said to herself, 'it was a little little shated of the sooe of the coomouse it was a little lad good to the little gooder head. and said to herself, 'it was a little little shated of the mouse of the good of the courte, and it was a little little shated in a little that the was a little little shated of the thmee said to see it was a little book of the was a little that she was so sereated to hare a little the began sitee of the was of the was a little that she was so seally and the sabbit was a little lad good to the little gooder head of the gad seared to see it was a little lad good to the little good''

Done.

#### Interpreting the Result Obtained ####

The first paragraph represent the generated seed text and the next paragraph is the generated text with the seed.

Upon observing the output/generated text we see that apart from general spelling mistakes like- "see" , but comparatively to the text generated with the smaller neural network here we are able to obtain more sensible and realistic model.

But still it seems quite a bit unsensical giving us a chance to further improve our model by enhancing our model, one way of doing that can be increasing the number of epochs, reducing the batch size and adopting more better strategies for model development.

The result still seem to be quite impressive and hence the project achieves the goal of generating new and unique text based upon a randomly chosen seed





## Following are the further Ideas which can be used in the working of our model ##

* Predicting fewer than 1,000 characters as output for a given seed.
* Remove all punctuation from the source text, and therefore from the models’ vocabulary.
* Try a one hot encoding for the input sequences.
* Train the model on padded sentences rather than random sequences of characters.
* Add more memory units to the layers and/or more layers.
* Experiment with scale factors (temperature) when interpreting the prediction probabilities.
* Changing the LSTM layers to be “stateful” to maintain state across batches.


## Resources and Referrences Used ##

This character text model is a very popular way for generating text using recurrent neural networks.
Below are some resources which were used in developing this project

--Generating Text with Recurrent Neural Networks [pdf], 2011 (http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)
--MXNet tutorial for using an LSTM for text generation. (http://mxnetjl.readthedocs.io/en/latest/tutorial/char-lstm.html)
--Keras code example of LSTM for text generation. (https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py)
-- Text Genration in python with Keras (https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/)
--Lasagne code example of LSTM for text generation. (https://github.com/Lasagne/Recipes/blob/master/examples/lstm_text_generation.py)


## Summary/Conclusion ##

In this project we could discover and could efficiently develop and train a LSTM recurrent neural network model for text generation with the Keras.

The text generated is unique and is based upon the randomly chosen seed.

1) These 3 goals and learnings were primarily have been achieved after implementing the first part of our project:
* How to prepare text for developing a word-based language model.
* How to design and fit a neural language model with a learned embedding and an LSTM hidden layer.
* How to use the learned language model to generate new text with similar statistical properties as the source text.

2) These 3 goals and learnings were primarily have been achieved after implementing the second part of our project:
* How to train an LSTM network on text sequences
* How to use the trained network to generate new sequences.
* How to develop stacked LSTM networks and lift the performance of the model.


