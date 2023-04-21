import random
import json

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow import keras
from keras.models import load_model


lemmatizer = WordNetLemmatizer()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_params():
    intents = load_json("intents.json")
    words = load_json("words.json")
    classes = load_json("classes.json")
    model = load_model("questions.model")
    return intents, words, classes, model


def cleanup_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = cleanup_sentence(sentence)
    bag = [0] * len(words)
    for word in sentence_words:
        for i, w in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_intent(sentence, words, classes, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    err_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > err_threshold]

    results.sort(key=lambda x: x[1], reverse=True)
    result = [{
        "intent": classes[r[0]],
        "probability": str(r[1])
    } for r in results]

    return result


def get_response(user_intents, intents):
    tag = user_intents[0]["intent"]
    print(user_intents[0]["probability"])

    for intent in intents:
        if intent["tag"] == tag:
            result = random.choice(intent["responses"])
    return result


if __name__ == "__main__":
    intents, words, classes, model = get_params()
    while True:
        message = input("ask here: ")
        intents_from_user = predict_intent(message, words, classes, model)
        res = get_response(intents_from_user, intents)
        print(res)
