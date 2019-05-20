#!/usr/bin/env python
# coding: utf-8

# In[16]:


import random
import pronouncing

# Create reverse N-grams from a list of tokens
def reverseNgrams(tokens, n):
    ngrams = []
    for i in range(len(tokens)-1, 0+n-2, -1):
        ngram = []
        for j in range(i, i-n, -1):
            ngram.append(tokens[j])
        ngrams.append(ngram)
    return ngrams

# Organize N-grams in a frequency lookup table (N-layer nested dictionaries)
def setupModel(ngrams):
    lookup = {}
    n = len(ngrams[0])
    for ngram in ngrams:
        ptr = lookup
        for i in range(0, n):
            if i == n-1:
                ptr.setdefault(ngram[i], 0)
                ptr[ngram[i]] += 1
            else:
                try: ptr = ptr[ngram[i]]
                except KeyError: 
                    ptr.setdefault(ngram[i], {})
                    ptr = ptr[ngram[i]]
    return lookup

# Loads all first words into an array for efficiency
def getFirstWords(corpus):
    firstWords = []
    for first in corpus:
        firstWords.append(first)
    return firstWords

# Randomly chooses a word from the corpus
def findFirst(corpus, firstWords):
    pick = random.randrange(0, len(firstWords)-1)
    return firstWords[pick]

# Randomly chooses a second word from the corpus based on the first 
def findSecond(first, corpus, firstWords):
    words = []
    try:
        for second in corpus[first]:
            words.append(second)
    except KeyError:
        return findFirst(corpus, firstWords)
        
    if len(words) == 0: return findFirst(corpus, firstWords)
    elif len(words) == 1: return words[0]
    
    pick = random.randrange(0, len(words)-1)
    return words[pick]
   
# Randomly chooses a third word from the corpus based on the first and second    
def findThird(first, second, corpus, firstWords):
    words = []
    try: 
        for third in corpus[first][second]:
            words.append(third)
    except KeyError:
        return findSecond(second, corpus, firstWords)
            
    if len(words) == 0: return findSecond(second, corpus, firstWords)
    elif len(words) == 1: return words[0]
    
    pick = random.randrange(0, len(words)-1)
    return words[pick]

# Builds sentences word by word
def addWord(sentence, first, second, corpus, firstWords):
    third = findThird(first, second, corpus, firstWords)
    sentence.append(third)
    first, second = second, third
    return first, second

# Randomly chooses a rhyming word from the corpus
def findRhyme(word, corpus):
    rhymes = pronouncing.rhymes(word)
    while (True):
        if len(rhymes) == 0:
            return None
      
        elif len(rhymes) == 1:
            try:
                corpus[rhymes[0]]
                return rhymes[0]
            except KeyError:
                return None        
        
        else:
            pick = random.randrange(0, len(rhymes)-1)
            try:
                corpus[rhymes[pick]]
                return rhymes[pick]
            except KeyError:
                rhymes.remove(rhymes[pick])
                continue

# Implementation of a couplet - AABB CCDD EEFF GGHH
def generateCouplet(corpus, lines, wordsPerLine):
    firstWords = getFirstWords(corpus)
    poem = []   
    for i in xrange(lines):            
        line = []  
        if i % 2 == 0:
            while (True):
                A = findFirst(corpus, firstWords)
                AA = findRhyme(A, corpus)                
                if AA != None:
                    break 
    
            first = A
            second = findSecond(first, corpus, firstWords)
            line += [first, second]
        
        if i % 2 == 1:
            first = AA
            second = findSecond(first, corpus, firstWords)
            line += [first, second]
        
        for j in xrange(wordsPerLine-2):
            first, second = addWord(line, first, second, corpus, firstWords)
        
        poem.append(line[::-1])
    return poem
 
# Handles capitalization lost from pre-processing
def poemProcessing(poem):
    for line in poem:
        line[0] = line[0][0].upper()+line[0][1:]
        for index, word in enumerate(line):
            if word[0:2] == "i'":
                line[index] = word[0:2].upper()+word[2:]
            if word == 'i':
                line[index] = word.upper()

# Prints poem line by line
def printPoem(poem):
    for line in poem:
        for word in line:
            print (word),
        print


# In[17]:



import json

# The input is simply a list of tokens or words from your text
# Tokens should be in order of appearance and collected per sentence
# Ideally, these will be tokens that have already been processed
# To lowercase, strip punctuation, remove numbers etc.
# I leave processing to the user because each text has specific needs 
sentence = ['w1','w2','w3','w4','w5','w6','w7','w8','w9']

# As you can see, you're generating "reverse" N-grams
# We can then build strings of words from the end, rather than the beginning
# This property is useful when you want to build sentences that rhyme
ngrams = reverseNgrams(sentence, 3)
for ngram in ngrams:
    print (ngram)


# In[18]:


model = setupModel(ngrams)
for row in model:
    print (row, model[row])


# In[45]:


with open('example.json', 'w') as outfile:
    json.dump(model, outfile)


# In[46]:


# def processToken(doc):
# 	# replace '--' with a space ' '
#     doc = doc.replace('--', ' ')
# 	# split into tokens by white space
# 	tokens = doc.split()
# 	# remove punctuation from each token
# 	table = str.maketrans('', '', string.punctuation)
# 	tokens = [w.translate(table) for w in tokens]
# 	# remove remaining tokens that are not alphabetic
# 	tokens = [word for word in tokens if word.isalpha()]
# 	# make lower case
# 	tokens = [word.lower() for word in tokens]
# 	return tokens
# # in_filename = 'Data/notorious-big.txt'

# tokens = clean_doc(sentences)


# In[51]:




from nltk.corpus import gutenberg

import json

# In this example I'm using a corpus from NLTK - Gutenburg Project
# Sara Bryant - Stories to Tell to Children

sentences = gutenberg.sents('bryant-stories.txt')
# Process text and collect reverse N-grams sentence by sentence
# Do not do this word by word or you'll have incoherent N-grams that span sentences

# in_filename = 'Data/notorious-big.txt'
# doc = load_doc(in_filename)
# print(doc[:200])
 
# # clean document
# tokens = clean_doc(doc)
# print(sentences)
# def processToken

ngrams = []
for sentence in sentences:
#     print(sentence)
#     tokens = processToken(sentence)
    ngrams += reverseNgrams(sentence, 3)

model = setupModel(ngrams)

with open('bryant-stories.json', 'w') as outfile:
    json.dump(model, outfile)


# In[53]:


# from Poesy import generateCouplet, poemProcessing, printPoem
import json
# artist_file = 'Data/notorious-big.txt'
# with open(artist_file) as f:
with open('bryant-stories.json', 'r') as infile:    
    corpus = json.load(infile)
  
# This will generate a couplet - AABB CCDD EEFF GGHH
# With 8 lines, 10 words each line
poem = generateCouplet(corpus, 8, 10)

# Handles capitilization and formatting
poemProcessing(poem) 

# Short function to print to console
printPoem(poem)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




