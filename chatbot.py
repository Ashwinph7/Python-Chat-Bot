import nltk
# nltk.download()
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

words = []
labels = []
docs_x = []       #to store the intents
docs_y = []		  #to store the tag for respective intent

#loading the data from the json file

with open("intents.json") as file:
	data = json.load(file)

try:
	with open("data.pickle","rb") as f:
		words,labels,training,output = pickle.load(f)

except:

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

	for x,doc in enumerate(docs_x):
		bag = []

		wrds= [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)


		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1 

		training.append(bag)
		output.append(output_row)

		with open("data.pickle","wb") as f:
			pickle.dump((words,labels,training,output),f)

training = numpy.array(training)
output = numpy.array(output)

# tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)


model = tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:
	model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
	model.save("model.tflearn")


def bag_of_words(s,words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]

	for s in s_words:
		for i,w in enumerate(words):
			if w == s:
				bag[i] = 1

	return numpy.array(bag)



def chat():
	print("Welcome to the ChatBot, please ask your question! To exit enter quit")

	while True:
		inpu = input(" You :")
		if inpu.lower() == "quit":
			break

		result = model.predict([bag_of_words(inpu,words)])[0]
		results_index = numpy.argmax(result)

		tag = labels[results_index]

		if result[results_index] > 0.70:

			for t in data["intents"]:
				if t['tag'] == tag:
					responses = t['responses']

			print("Bot :"+ random.choice(responses))
		else:

			print("Bot : I am not trained on that, please ask another question!")

chat()


