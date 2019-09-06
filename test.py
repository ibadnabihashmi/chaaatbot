import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

data = pickle.load( open( "katana-assistant-data.pkl", "rb" ) )
words = data['words']
classes = data['classes']

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

p = bow("Load bood pessure for patient", words)
print (p)
print (classes)

global graph
graph = tf.get_default_graph()

with open(f'katana-assistant-model.pkl', 'rb') as f:
    model = pickle.load(f)

def classify_local(sentence):
    ERROR_THRESHOLD = 0.25
    
    input_data = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    results = model.predict([input_data])[0]
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], str(r[1])))
    
    return return_list

print(classify_local('Ich habe mein Kopflicht gebrochen'))

