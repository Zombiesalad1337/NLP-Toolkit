from logging.config import stopListening 
import time
from tracemalloc import stop
start_time = time.time()
import numpy as np
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


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
from random import choice
from cgitb import enable
import PySimpleGUI as sg

import prediction
import summary

stop = model =  tokenizer = max_seq_len = None



print("Modules loaded in " + str(time.time() - start_time))
start_time = time.time()
# sg.theme_previewer()

sg.change_look_and_feel('DarkGreen3')
main_column = [[
    sg.Text("Welcome to NLP Suite"),
    sg.Button("Sentiment Analysis", size=(40, 20), enable_events=True, key="-SA-"),
    sg.Button("Summarisation", size=(40, 20), enable_events=True, key="-SUM-"),

]]




sa_column = [[sg.Multiline(size=(70,30), enable_events=True, key="-SA_INPUT-"),
            sg.Text(size=(20,5), enable_events=True, key="-SA_OUTPUT-", justification='center', k='-C-', font=("Arial", 32)),
            sg.Button("Back", size=(20,10), enable_events=True, key="-SA_BACK-"),]]

# main_layout = [ 
#     [
#         sg.Column(main_column)
#     ]
# ]

main_layout = [[i] for i in main_column[0]]

# sa_layout = [ 
#     [
#         sg.Column(sa_column)
#     ]
# ]


sa_layout = [[i] for i in sa_column[0]]


sum_column = [
            [sg.Text("Article File"),
            sg.In(size=(25, 1), enable_events=True, key="-SUM_FILE-"),
            sg.FileBrowse(),],
            [
                sg.Multiline(size=(70, 30), key="-SUM_OUTPUT-"),
            ],
            [
                sg.Button("Back",size=(10,5), enable_events=True, key="-SUM_BACK-"),
            ]

            ]
# sa_loading_layout = [
#     [sg.Text("Please wait while the model loads", size=(50, 20), enable_events=True, key="-SA-LOAD-")]
# ]


sum_layout = sum_column

window = sg.Window("NLP Suite", main_layout, finalize=True)

window_sa = sg.Window("Sentiment Analysis", sa_layout, finalize=True)

window_sum = sg.Window("Text Summarization", sum_layout, finalize=True)

window_sa.hide()
window_sum.hide()

count_sa = 0

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        window_sa.close()
        break

    if event == "-SA-":
        count_sa += 1
        window.hide()
        window_sa.un_hide()

        if count_sa == 1:
            # window_sa_load = sg.Window("Please wait...", sa_loading_layout)
            # window_sa.hide()
            stop = prediction.stop_words() 
            max_seq_len, tokenizer, model = prediction.__setup__()
            # window_sa.un_hide()
            # window_sa_load.close()



        while True:
            event_sa, values_sa = window_sa.read()
            if event_sa == "Exit" or event_sa == sg.WIN_CLOSED:
                break
            
            if event_sa == "-SA_INPUT-":
                # window_sa["-SA_OUTPUT-"].update(choice(["bad", "good"]))
                input_text = values_sa["-SA_INPUT-"]
                if len(input_text) > 0:
                    if input_text[-1] in [' ', '.', ',']:
                        input_text = input_text[: -1]
                        print(input_text)

                        val = prediction.predict_sentiment(input_text, stop, max_seq_len, tokenizer, model)
                        print(val)
                        if (val >= 0.8):
                            window_sa["-SA_OUTPUT-"].update("Very Positive!")
                        elif (val >= 0.45):
                            window_sa["-SA_OUTPUT-"].update("Positive")
                        elif (val >= 0.34):
                            window_sa["-SA_OUTPUT-"].update("Mixed")
                        elif (val >= 0.2):
                            window_sa["-SA_OUTPUT-"].update("Negative")
                        else:
                            window_sa["-SA_OUTPUT-"].update("Very Negative")
            elif event_sa == "-SA_OUTPUT-":
                window_sa["-SA_OUTPUT-"].update("Inputs in the input field!")
            elif event_sa == "-SA_BACK-":
                #perform cleaning
                window_sa["-SA_INPUT-"].update("") 
                window_sa["-SA_OUTPUT-"].update("")
                break
        window_sa.hide()
        window.un_hide( )
        
    elif event == "-SUM-":
        while True:
            window.hide()
            window_sum.un_hide()
            event_sum, values_sum = window_sum.read()
            
            if event_sum == "Exit" or event_sum == sg.WIN_CLOSED:
                break
            
            if event_sum == "-SUM_FILE-":
                file = values_sum["-SUM_FILE-"]
                summarized = summary.generate_summary(file)
                window_sum["-SUM_OUTPUT-"].update(summarized) 
            elif event_sum == "-SUM_BACK-":
                window_sum["-SUM_FILE-"].update("")
                window_sum["-SUM_OUTPUT-"].update("")
                break
        window_sum.hide()
        window.un_hide()



window.close()

