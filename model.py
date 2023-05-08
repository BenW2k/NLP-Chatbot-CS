# Import dependencieS
import nltk
import numpy as np
import pandas
import random
import string
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import secret for data file path prefix
import secret

# Access data through directory
data_path = os.path.join(secret.PATH_PREFIX, 'data', 'data.txt')
f = open(data_path)

# Set variable for data
raw_data = f.read()

# Variable to store the bot name (can be changed to better represent its role as a chatbot)
bot_name = 'BenBot'
# Variable to store bot topic
bot_topic = 'Ben Enterprises'

# Data cleaning

# Converts text to lower case format
data = raw_data.lower()
# Splits data into sentence tokens
sent_tokens = nltk.sent_tokenize(data)
# Splits data in word tokens
word_tokens = nltk.word_tokenize(data)

# Breaking words down into their root meanings

lemm = nltk.stem.WordNetLemmatizer()

def lemmatize_tokens(tokens):
    lemm = nltk.stem.WordNetLemmatizer()
    return [lemm.lemmatize(token)for token in tokens]

# Formatting text in a standardised form
def normalize_tokens(text):
    remove_punc = dict((ord(punct), None) for punct in string.punctuation)
    return lemmatize_tokens(nltk.word_tokenize(text.lower().translate(remove_punc)))

# Defining how the bot will greet the user
greet_inputs = ('hi', 'hello', 'hey', 'yo')
greet_responses  = ('Hello! How can I help you today?', 'Hi! How can I help you today?', "what's up? How can I help you today?")

# Function to greet user if greeted first
def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# Function to determine the bot's response
def bot_response(user_response):
    bot_response = ''
    TfidfVec = TfidfVectorizer(tokenizer= normalize_tokens, stop_words= 'english') # Performs tokenization on the text whilst removing english stopwords
    TfidFit = TfidfVec.fit_transform(sent_tokens) # Fit and transforming TfidVectors
    vals = cosine_similarity(TfidFit[-1], TfidFit) # Performing cosine similarity, 
    idx = vals.argsort()[0][-2] # Uses argsort to find the most similar token
    flat = vals.flatten() # Flattens vals into a one dimensional array
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        bot_response = bot_response + "I'm sorry, I do not understand, please try again."
        return bot_response
    else:
        bot_response = bot_response + sent_tokens[idx]
        return bot_response
    

flag = True
print(f"{bot_name}: Hello, I am {bot_name}, what about {bot_topic} would you like to know today? \n If you would like to exit, please reply with 'goodbye'")

while (flag == True):
    user_response = input()
    user_response = user_response.lower()
    if (user_response != 'goodbye'):
        if(user_response == 'thank you' or user_response == 'thanks'):
            flag = False
            print(f'{bot_name}: You are welcome')
        else:
            if (greet(user_response) != None):
                print(f'{bot_name}: ' + greet(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens = word_tokens + nltk.word_tokenize(user_response)
                final_words = list(set(word_tokens))
                print(f'{bot_name}: ' + bot_response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print(f'{bot_name}: Goodbye!')     
