#!/usr/bin/env python
# coding: utf-8

# In[19]:


# ngram=2


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
import nltk
nltk.download('stopwords',quiet=True)
nltk.download('punkt',quiet=True)

import nltk
import os
import string
import numpy as np
import copy
import pandas as pd
import pickle
import re
import math
import threading
import time
import sys
from nltk.util import ngrams


import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State# Load Data
import sklearn
import base64
from google_trans_new import google_translator  
translator = google_translator()






tf_idf = {}
f1=open('dataset_final.pkl','rb')
dataset = pickle.load(f1)



tf_idf1 = pickle.load(open("tfidf_unigram_final.pkl",'rb'))
tf_idf2 = pickle.load(open("tfidf_bigram_final.pkl",'rb'))
tf_idf3 = pickle.load(open("tfidf_trigram_final.pkl",'rb'))




def load_folders(path):
    folders = [x[0] for x in os.walk(path)]
    folders[0] = folders[0][:len(folders[0])-1]
    folders=folders[1:]
    return folders

folders = load_folders(path=str(os.getcwd())+'/bbc/')
path_bbc = folders[0][:folders[0].index("bbc")+3]

def print_doc(id):
    print(dataset[id])
    path = folders[0][:folders[0].index("bbc")+3]
    file = open(path+dataset[id][0], 'r',encoding="utf8", errors='ignore')
    text = file.read().strip()
    file.close()
    return(text)

# Preprocessing
def convert_lower_case(data):
    return np.char.lower(data)

def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text

def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data

def remove_apostrophe(data):
    return np.char.replace(data, "'", "")

def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text

def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text

def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data) #remove comma seperately
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = convert_numbers(data)
    data = stemming(data)
    data = remove_punctuation(data)
    data = convert_numbers(data)
    data = stemming(data) #needed again as we need to stem the words
    data = remove_punctuation(data) #needed again as num2word is giving few hypens and commas fourty-one
    data = remove_stop_words(data) #needed again as num2word is giving stop words 101 - one hundred and one
    return data

#number of documents word occured
def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


def generate_ngrams(s, n):
    tokens = [token for token in s.split(" ") if token != ""]
    output = list(ngrams(tokens, n))
    return output
    
def matching_score(k, query, ngrams):
    preprocessed_query = preprocess(query)
    tf_idf = {}
    if(ngrams ==1):
        tf_idf = tf_idf1
    elif(ngrams ==2):
        tf_idf = tf_idf2
    elif(ngrams ==3):
        tf_idf = tf_idf3
    
    if(len(preprocessed_query.split())<ngrams):
        print("Enter sentence with more token for ngram = "+ str(ngrams))
    else:
        tokens = generate_ngrams(preprocessed_query, n=ngrams)
        print("Matching Score")
      #print("\nQuery:", query)
      #print("")
        print(tokens)

        query_weights = {}

        for key in tf_idf:
            if key[1] in tokens:
                try:
                    query_weights[key[0]] += tf_idf[key]
                except:
                    query_weights[key[0]] = tf_idf[key]

        query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

        l = []
        for i in query_weights[:k]:
            l.append(i[0])
        print("Total number of documents retrieved: "+str(len(l)))
        print(l[0:20])
        return l
    
def calculate_TFIDF_query(query, ngrams):
    preprocessed_query = preprocess(query)
    query_tokens = generate_ngrams(preprocessed_query, n=ngrams)
    if(len(query_tokens) ==0):
        return {"invalid":"invalid"}
    print(query_tokens)
    tf_idf_query = {}
    tokens = query_tokens
    counter = Counter(tokens)
    words_count = len(tokens)
    for token in set(tokens):
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1)) #numerator is added 1 to avoid negative values
        tf_idf_query[token] = tf*idf
    return tf_idf_query
  
def sparse_cosine_sim(tf_idf_query, tf_idf):
    cosine_sim = {}
    for i in range(N):
        for key in tf_idf_query:
            try:
                prod_ab = tf_idf[(i, key)]*tf_idf_query[key] 
                try:
                    cosine_sim[i] += prod_ab
                except:
                    cosine_sim[i] = prod_ab
            except:
                pass
    for key in cosine_sim:
        doc_id = key
        sq_tf_idf = 0
        for key in tf_idf:
            if key[0]== doc_id:
                sq_tf_idf += tf_idf[key]**2
        cosine_sim[doc_id] /= np.sqrt(sq_tf_idf)

    return cosine_sim
  
def get_cos_sim_klist(k, query, ngrams):
    print("Cosine Similarity")
    tf_idf = {}
    if(ngrams ==1):
        tf_idf = tf_idf1
    elif(ngrams ==2):
        tf_idf = tf_idf2
    elif(ngrams ==3):
        tf_idf = tf_idf3
    
    tf_idf_query = calculate_TFIDF_query(query, ngrams)
    if(list(tf_idf_query.keys())[0] == 'invalid'):
        return ["invalid",ngrams]
    cosine_sim = sparse_cosine_sim(tf_idf_query, tf_idf)
    cosine_sim = sorted(cosine_sim.items(), key=lambda x: x[1], reverse=True)
    print("Total ", len(cosine_sim), " documents found.\nTop ", min(k, len(cosine_sim)), " results are: ")

    l = []
    for i in cosine_sim[:k]:
        l.append(i[0])
    print("Total number of documents retrieved: "+str(len(l)))
    print(l[0:20])
    return l
  





N=len(dataset)
















app = JupyterDash(__name__)
val="Search Query"
options1 = []
options2 = []
gram = [1,2,3]
methods= ['tfidf', 'cosine similarity']

app.layout = html.Div(children = [
      html.Div(children = [
        html.H2("National Institute of technology, Silchar",
                style= {'textAlign' : 'center', 
                        'backgroundColor' : '#1f7370',
                        'borderRadius' : 5,
                        'boxShadow': '0px 5px 5px #888',
                        'color' : 'white',
                        'top' : '5%'}),
        html.H4("Final Year Project By:",
                style = {'textAlign' : 'center'}),
        html.Div(children = [
          html.P("Shiv Warsi"),
          html.P("Ahamed Md. Hussain"),
          html.P("Anurag Kumar"),
          html.P("Bairu Mukesh Goud"),
        ], 
        style= {'textAlign' : 'center',
                'paddong' : '0%'}),
      ], 
      style = {'borderRadius' : 10,
               'boxShadow': '0px 5px 5px #888',
               'padding' : '0.7%',
               'border':'3px solid #1f7370',
               'backgroundColor' : '#f3f5f2'
              }),

      html.Div(children = [
        html.H2("Cross Lingual Information Retrieval",
                style= {'textAlign' : 'center',
                        'color' : 'white',
                        'borderRadius' : 5,
                        'boxShadow': '0px 5px 5px #888',
                        'backgroundColor' : '#1f7370'}),
        html.H3("Enter value for ngrams.",
                style = {'textAlign' : 'center'}),
        html.Div(
            children = [dcc.Dropdown(
                        id='dropdown1',
                        options=[{'label': g, 'value': g} for g in gram],
                        value=1
                        )], 
                        style={'textAlign' : 'center',
                               'borderRadius' : 2}),
        html.H3("Choose method.",
                style = {'textAlign' : 'center'}),
        html.Div(
            children = [dcc.Dropdown(
                        id='dropdown2',
                        options=[{'label': m, 'value': m} for m in methods],
                        value='tfidf'
                        )], 
                        style={'textAlign' : 'center',
                               'borderRadius' : 2}),
        html.H3("Enter the query you want to search the documnet for.",
                style = {'textAlign' : 'center'}),
        html.Div(
            children = [dcc.Input(id='my-id', value=val, type="text", size="50")], 
                        style={'textAlign' : 'center',
                               'borderRadius' : 2}),
        html.Br(),
        html.Br(),
        html.Button('Search', id='button', 
                    style= {'textColor' : 'white',
                            'background-color' : '#1f7370',
                            'borderRadius' : 2,
                            'border' : 'none',
                            'height' : '30px',
                            'width' : '90px',
                            'boxShadow': '0px 5px 5px #888',
                            'borderRadius' : 2,
                            'color' : '#000000', 
                            'text-align' : 'center', 
                            'left' : '47%', 
                            'top' : '30%'}) 
        ], 
        style = {'textAlign' : 'center',
                 'borderRadius' : 10,
                 'boxShadow': '0px 5px 5px #888',
                 'padding' : '2%',
                 'margin' : '3%',
                 'border':'3px solid #1f7370',
                 'backgroundColor' : '#f3f5f2'
                }),

      html.Div(id = 'my-gram', children = []),
    html.Div(id = 'my-method', children = []),
    html.Div(id = 'my-div', children = []),
])

#render
@app.callback(
    Output(component_id='my-gram', component_property='children'),
    [Input('button', 'n_clicks'),Input('dropdown1', 'value')]
)
def update_gram(n_clicks, value):
    options1.append('{}'.format(value))
    
@app.callback(
    Output(component_id='my-method', component_property='children'),
    [Input('button', 'n_clicks'),Input('dropdown2', 'value')]
)
def update_method(n_clicks, value):
    options2.append('{}'.format(value))
    
@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input('button', 'n_clicks')],
    [State(component_id='my-id', component_property='value')]
)

def update_output_div(n_clicks, value):
    if value == val : 
        return ""
    else :
        tf_idf={}
        docs = [];
        titles=[];
        text = translator.translate(value, lang_src='hi', lang_tgt='en')
        Q=[];
        
        if(options1[-1]==1):
            tf_idf = pickle.load(open("tfidf_unigram_final.pkl",'rb'))
        elif(options1[-1]==2):
            tf_idf = pickle.load(open("tfidf_bigram_final.pkl",'rb'))
        elif(options1[-1]==3):
            tf_idf = pickle.load(open("tfidf_trigram_final.pkl",'rb'))
            
            
            
        
        
        if options2[-1] == 'tfidf':
            Q = matching_score(2225, text, int(options1[-1]))
            if Q is None:
                docs.append(html.Div(children = [html.H4('Enter more tokens for ngram = '+str(int(options1[-1])))], 
                      style = {'textAlign' : 'justify',
                 'borderRadius' : 5,
                 'boxShadow': '0px 5px 5px #888',
                 'padding' : '1%',
                 'margin' : '3%',
                 'border':'3px solid #1f7370',
                 'backgroundColor' : '#f3f5f2'
                }))
            else:
                length = len(Q)
                top20 = length if length <20 else 20
                docs.append(html.Div(children = [html.H4('Total number of documents retrieved : ' + str(length)),
                                                 html.H4('Top '+ str(top20)+ ' Documents are : ')
                                            ], 
                      style = {'textAlign' : 'justify',
                 'borderRadius' : 5,
                 'boxShadow': '0px 5px 5px #888',
                 'padding' : '1%',
                 'margin' : '3%',
                 'border':'3px solid #1f7370',
                 'backgroundColor' : '#f3f5f2'
                }))

                for i in Q[0:20]:
                    item = print_doc(i)
                    index = item.find("\n\n");
                    title = item[0:index]
                    body = item[index+2:len(item)]
                    docs.append(html.Div(children = [html.H4('Title : ' + title),
                                             html.H5('Body : ', style={'display' : 'inline-block'}),
                                             html.P(body, style={'display' : 'inline-block'})
                                                ], 
                          style={'text-color' : '#ff0000',
                         'borderRadius' : 2,
                         'boxShadow': '0px 5px 5px #888',
                         'padding' : '2%',
                         'margin' : '3%',
                         'textAlign' : 'justify',
                         'border':'3px solid #1f7370',
                         'backgroundColor' : '#f3f5f2'
                                }))
            return docs
        elif options2[-1] == 'cosine similarity':
            Q = get_cos_sim_klist(2225, text, int(options1[-1]))
            if('invalid' in Q):                
                docs.append(html.Div(children = [html.H4('Enter more tokens for ngram = '+str(Q[1]))], 
                      style = {'textAlign' : 'justify',
                 'borderRadius' : 5,
                 'boxShadow': '0px 5px 5px #888',
                 'padding' : '1%',
                 'margin' : '3%',
                 'border':'3px solid #1f7370',
                 'backgroundColor' : '#f3f5f2'
                }))
            else:
                length = len(Q)
                top20 = length if length <20 else 20
                docs.append(html.Div(children = [html.H4('Total number of documents retrieved : ' + str(length)),
                                                 html.H4('Top '+ str(top20)+ ' Documents are : ')
                                            ], 
                      style = {'textAlign' : 'justify',
                 'borderRadius' : 5,
                 'boxShadow': '0px 5px 5px #888',
                 'padding' : '1%',
                 'margin' : '3%',
                 'border':'3px solid #1f7370',
                 'backgroundColor' : '#f3f5f2'
                }))

                for i in Q[0:20]:
                    item = print_doc(i)
                    index = item.find("\n\n");
                    title = item[0:index]
                    body = item[index+2:len(item)]
                    docs.append(html.Div(children = [html.H4('Title : ' + title),
                                             html.H5('Body : ', style={'display' : 'inline-block'}),
                                             html.P(body, style={'display' : 'inline-block'})
                                                ], 
                          style={'text-color' : '#ff0000',
                         'borderRadius' : 2,
                         'boxShadow': '0px 5px 5px #888',
                         'padding' : '2%',
                         'margin' : '3%',
                         'textAlign' : 'justify',
                         'border':'3px solid #1f7370',
                         'backgroundColor' : '#f3f5f2'
                                }))
            return docs

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:


# !ngrok authtoken 1rJ3IkEXHXSu0cuF0XUD2ViEqnc_3cBpo4be9Pbd1krKTDecE


# In[ ]:


# from pyngrok import ngrok

# Open a HTTP tunnel on the default port 80
# public_url = ngrok.connect()


# In[ ]:


# public_url


# In[ ]:


# ngrok.kill()


# In[ ]:




