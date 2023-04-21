import os
import json
import random

import numpy as np
import pymongo
from dotenv import load_dotenv
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD


load_dotenv()
db = pymongo.MongoClient(os.getenv("MONGO_URL")).questions_task.questions

from keras import backend 
print("GPUs Available: ", backend._get_available_gpus())

lemmatizer = WordNetLemmatizer()
nltk.download("punkt")
nltk.download("wordnet")


def dump_to_json():
    data = []
    tags = set(question["theme"] for question in db.find({}, {"theme": 1}))
    for tag in tags:
        data_by_tag = [i for i in db.find({"theme": tag})]
        questions = [item["question"] for item in data_by_tag if len(str(item["question"])) > 3]
        answers = [item["answer"] for item in data_by_tag if len(str(item["answer"])) > 3]
        data.append({
            "tag": tag,
            "patterns": questions,
            "responses": answers
        })
    with open("intents.json", "w") as f:
        json.dump(data, f, indent=4)


def collect():
    with open("intents.json", "r") as f:
        intents = json.load(f)
        
    words = []
    classes = set()
    documents = []
    ignore_words = [",", ".", "!", "?", "-", "/", "\\", "здравствуйте", "коллеги"]

    for intent in intents:
        for pattern, response in zip(intent["patterns"], intent["responses"]):
            tokens_pattern = nltk.word_tokenize(pattern)
            tokens_response = nltk.word_tokenize(response)
            tokens = tokens_pattern + tokens_response
            words.extend(tokens)
            documents.append((tokens, intent["tag"], pattern, response))
            classes.add(intent["tag"])
    
    words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_words]
    words = sorted(set(words))
    classes = sorted(classes)

    with open("classes.json", "w") as f:
        json.dump(classes, f, indent=4)
    with open("words.json", "w") as f:
        json.dump(words, f, indent=4)
    
    return words, classes, documents
    
    
def train(words, classes, documents):
    training = []
    output = [0] * len(classes)    

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
            
        output_row = list(output)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])
        
    random.shuffle(training)
    training = np.array(training)
    
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    
    model = Sequential()
    print("created model")
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    print("added dense")
    model.add(Dropout(0.5))
    print("added dropout")
    model.add(Dense(64, activation="relu"))
    print("added dense 2")
    model.add(Dropout(0.5))
    print("added dropout 2")
    model.add(Dense(len(train_y[0]), activation="softmax"))
    print("added dense 3")
    
    sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    print("created sgd")
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    print("compiled model")

    try:
        model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    except Exception as _:
        ...
    finally:
        print("model fitted")
        model.save("questions.model")
        print("model saved")
    
    
if __name__ == "__main__":
    dump_to_json()
    args = collect()
    train(*args)
    