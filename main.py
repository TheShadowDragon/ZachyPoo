import json
import nltk
import numpy as np
import random
import tflearn as tf
import pickle
import os
import discord
import asyncio
from discord.ext import commands
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')

client = discord.Client()
client = commands.Bot(command_prefix = '#')

Reply = None;

@client.event
async def onready():
  print("online!")
  
stemmer = LancasterStemmer()
  
with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
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
  
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
  
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
  
    for x, doc in enumerate(docs_x):
      bag = []
      wrds = [stemmer.stem(w.lower()) for w in doc]
  
      for w in words:
          if w in wrds:
              bag.append(1)
          else:
              bag.append(0)
  
      output_row = out_empty[:]
      output_row[labels.index(docs_y[x])] = 1
  
      training.append(bag)
      output.append(output_row)
  
  
    training = np.array(training)
    output = np.array(output)
  
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
  
net = tf.input_data(shape=[None, len(training[0])])
net = tf.fully_connected(net, 16)
net = tf.fully_connected(net, 16)
net = tf.fully_connected(net, len(output[0]), activation="softmax")
net = tf.regression(net)
  
model = tf.DNN(net)
  
model.fit(training, output, n_epoch=2500, batch_size=8, show_metric=True)
model.save("model.tf")
  
try:
    model.load("model.tf")
except:
    model.fit(training, output, n_epoch=2500, batch_size=8, show_metric=True)
    model.save("model.tf")
  
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
  
    for sentence in s_words:
        for i, w in enumerate(words):
            if w == sentence:
                bag[i] = 1
  
    return np.array(bag)

  
@client.command()
async def chat(ctx, *, args):
  results = model.predict([bag_of_words(args, words)])[0]
  results_index = np.argmax(results)
  tag = labels[results_index]
  for tg in data["intents"]:
    if tg['tag'] == tag:
      responses = tg['responses']
      
  await ctx.send(random.choice(responses))
client.run(os.getenv("SECRET")) 
