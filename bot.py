from flask import Flask,render_template,request,redirect,url_for, flash, session, jsonify
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer() 

import numpy as np
import tflearn
import tensorflow as tf
from tensorflow.python.framework import ops
import random
import json
import pickle
import pathlib

app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

with open("intents.json") as file:
    data = json.load(file)

words=[]
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w!="?"]
words = sorted(list(set(words)))

labels = list(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
    bag = []
    wrds=[stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])]=1

    training.append(bag)
    output.append(output_row) 

training = np.array(training)
output = np.array(output)


ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])]) #create a neuron layer followed by 2 others
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #give probability to o/p neurons
net = tflearn.regression(net)

model = tflearn.DNN(net) #create tensorflow neural net type

try:
    #t
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)#fit the model
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i,w in enumerate(words):
            if w == se:
                bag[i]=1

    return np.array(bag) 

@app.route("/bot_response")
def chat():
    #print("Start talking with marilee!")
    while True:
        inp = request.args.get('msg')
        results = model.predict([bag_of_words(inp,words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]
        
        if results[results_index]>0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            return jsonify(respo1=random.choice(responses))

        else:
            return jsonify(respo1="I did not get that! Please try again")

if __name__ == "__main__":
    app.run(debug=True)