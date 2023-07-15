from django.http import HttpResponse
from django.shortcuts import render
import os
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import json

def index(request):
    return render(request, 'index.html')

def chatbot(request):
    if request.method == "POST":
        user_input = request.POST.get('search')
        response = get_chatbot_response(user_input)
        result={'msg': user_input ,'response': response}
        return render(request, 'chatbot.html', result)
        

    return render(request, 'chatbot.html')

def get_chatbot_response(user_input):
    tokenized_words = nltk.word_tokenize(user_input)
    tokenized_words = [lemmatizer.lemmatize(w.lower()) for w in tokenized_words]

    input_data = np.array([0] * len(words))
    for w in tokenized_words:
        if w in words:
            input_data[words.index(w)] = 1

    results = model.predict(np.array([input_data]))
    results_index = np.argmax(results)
    tag = labels[results_index]

    for intent in data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']

    return random.choice(responses)

# Step 1: Reading and preprocessing the data
with open('./template/intents.json') as file:
    data = json.load(file)

lemmatizer = WordNetLemmatizer()
words = []
labels = []
docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokenized_words = nltk.word_tokenize(pattern)
        words.extend(tokenized_words)
        docs_x.append(tokenized_words)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w != '?']
words = sorted(list(set(words)))
labels = sorted(labels)

# Step 2: Creating training and testing data
training_data = []
output_data = []
out_empty = [0] * len(labels)

for x, doc in enumerate(docs_x):
    bag = []

    for w in words:
        if w in doc:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training_data.append(bag)
    output_data.append(output_row)

training_data = np.array(training_data)
output_data = np.array(output_data)

# Step 3: Creating the model
model = Sequential()
model.add(Dense(128, input_shape=(len(training_data[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(output_data[0]), activation='softmax'))

# Step 4: Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Training the model
model.fit(training_data, output_data, epochs=200, batch_size=5, verbose=1)

# Step 6: Saving the model
model.save('chatbot_model.h5')

# Step 7: Loading the model
model = tf.keras.models.load_model('chatbot_model.h5')

# Initialize the chatbot
# initialize()

# Running the Django web application
# ...
