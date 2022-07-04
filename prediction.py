import time
import numpy as np
import pickle
import pandas as pd
import warnings


from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re, string, unicodedata
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import transformers
from tensorflow.keras.models import load_model



def stop_words():
    # stop = set(stopwords.words('english'))
    stop = set()
    punctuations = list(string.punctuation)
    stop.update(punctuations)
    return stop

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

# removing url's
def remove_between_square_brackets(text):
    return re.sub(r'http\s+', '', text)

#removing the stopwords from text
def remove_stopwords(text, stop):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop and i.strip().lower().isalpha():
            final_text.append(i.strip().lower())
    return " ".join(final_text)

#removing the noisy text
def denoise_text(text, stop):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    text = remove_stopwords(text, stop)
    print(text)
    return text


def __setup__():
    max_seq_len = 225
    start_time = time.time()
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer loaded in " + str(time.time() - start_time))
    start_time = time.time()
    model = load_model('./checkpoints/')
    print("Model loaded in " + str(time.time() - start_time))
    return max_seq_len, tokenizer, model



def convert_input(text, stop,  max_seq_len, tokenizer):
    text = denoise_text(text, stop)
    text = [text]
    seq_test = tokenizer.texts_to_sequences(text)
    pad_test = pad_sequences(seq_test,truncating = 'post', padding = 'pre',maxlen=max_seq_len)
    return pad_test


def predict_sentiment(text, stop, max_seq_len, tokenizer, model):
    text = convert_input(text, stop, max_seq_len, tokenizer)
    val = model.predict(text)
    return val