!pip install num2words
!pip install google_trans_new
from google_trans_new import google_translator  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from num2words import num2words
import nltk
nltk.download('stopwords')
nltk.download('punkt')

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
import pickle
from nltk.util import ngrams

def load_folders(path):
    folders = [x[0] for x in os.walk(path)]
    folders[0] = folders[0][:len(folders[0])-1]
    folders=folders[1:]
    return folders
folders = load_folders(path=str(os.getcwd())+'/bbc/')

## Collecting the file names and titles
def total_documents(folder_list):
    length = 0
    for i in folder_list:
        length = length+ len(os.listdir(i))
    print("Total number of documents found: ", length)
    return length

def load_documents(folder_list):
    dataset=[]
    k=0;
    for i in folders:
        for x in os.listdir(i):
            f=open(os.path.join(i, x), 'r',encoding="utf8", errors='ignore')
            text = f.read().strip()
            f.close()

            title = text[:text.index('\n')].strip()
            path = os.path.join(i+"/"+x)

            dataset.append((path,title))
            k=k+1
    print("All documents loaded!")
    return dataset

def print_doc(id):
    print(dataset[id])
    file = open(dataset[id][0], 'r',encoding="utf8", errors='ignore')
    text = file.read().strip()
    file.close()
    print(text)

N = total_documents(folders)
dataset = load_documents(folders)

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
    data = convert_numbers(data)
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

## Extracting Data
def generate_ngrams(s, n):
    tokens = [token for token in s.split(" ") if token != ""]
    output = list(ngrams(tokens, n))
    return output

def extract_ntokens(length, ngrams=1,dataset=None):
    processed_text = []
    processed_title = []
    count=0
    print("Preprocessing...")
    for i in dataset[:length]:
        file = open(i[0], 'r', encoding="utf8", errors='ignore')
        text = file.read().strip()
        file.close()

        # removes special characters with ' '
        text = re.sub('[^a-zA-z\s]', '', text)
        text = re.sub('_', '', text)

        # Change any white space to one space
        text = re.sub('\s+', ' ', text)
        processed_text.append(generate_ngrams(preprocess(text), n=ngrams))
        processed_title.append(generate_ngrams(preprocess(i[1]), n=ngrams))
        count=count+1
    print("Completed!")
    return processed_text, processed_title

def print_tokens(doc_id, token_list):
    print("No. of tokens present in Document "+ str(doc_id) + " : ",len(token_list[doc_id]))
    print()
    for token in token_list[doc_id]:
        print(token, end=' ')

text_tokens, title_tokens = extract_ntokens(length=N,ngrams=2,dataset=dataset)


## Calculating DF for all words
def calculate_DF(num_doc, text_token, title_token):
    DF = {}
    for i in range(num_doc):
        tokens = text_token[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}
        tokens = title_token[i]
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}          
    for i in DF:
        DF[i] = len(DF[i])
    total_vocab_size = len(DF)
    total_vocab = [x for x in DF]
    return DF, total_vocab_size, total_vocab

DF, total_vocab_size, total_vocab = calculate_DF(N, text_tokens, title_tokens)
#now DF contains tokens and the number of docments the word is present

#number of documents word occured
def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


### Calculating TF-IDF for body, we will consider this as the actual tf-idf as we will add the title weight to this.
def calculate_TFIDF(num_docs, text_token, title_token):
    doc = 0
    tf_idf_text = {}
    for i in range(N):
        tokens = text_token[i]
        counter = Counter(tokens + title_token[i])
        words_count = len(tokens + title_token[i])
        for token in set(tokens):
            #freq of a word of a particular doc in that doc -> percentage
            tf = counter[token]/words_count
            #freq of that word in the whole corpus -> count
            df = doc_freq(token)
            idf = np.log((N+1)/(df+1))
            tf_idf_text[doc, token] = tf*idf
        doc += 1
    return tf_idf_text

tf_idf_text = calculate_TFIDF(N, text_tokens, title_tokens)


### Calculating TF-IDF for Title
def calculate_TFIDF_title(num_docs, text_token, title_token):
    doc = 0
    tf_idf_title = {}
    for i in range(N):
        tokens = title_token[i]
        counter = Counter(tokens + text_token[i])
        words_count = len(tokens + text_token[i])
        for token in set(tokens):
            tf = counter[token]/words_count
            df = doc_freq(token)
            idf = np.log((N+1)/(df+1)) #numerator is added 1 to avoid negative values
            tf_idf_title[doc, token] = tf*idf
        doc += 1
    return tf_idf_title


tf_idf_title = calculate_TFIDF_title(N, text_tokens, title_tokens)


## Merging the TF-IDF according to weights
def merge(text, title, alpha=0.25):
    for i in text:
        text[i] *= alpha
    for i in title:
        text[i] = title[i]
    return text

tf_idf = merge(tf_idf_text, tf_idf_title)


# TF-IDF Matching Score Ranking
def matching_score(k, query, ngrams=1):
    preprocessed_query = preprocess(query)
    tokens = generate_ngrams(preprocessed_query, n=ngrams)
    print("Matching Score")
    print("\nQuery:", query)
    print("")
    print(tokens)
    
    query_weights = {}

    for key in tf_idf:
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]
    
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    
    print("")
    l = []
    for i in query_weights[:k]:
        l.append(i[0])
    print(l)
    return l

#Cosine Similarity

def calculate_TFIDF_query(query, ngrams=2):
  preprocessed_query = preprocess(query)
  query_tokens = generate_ngrams(preprocessed_query, n=ngrams)
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
  tf_idf_query = calculate_TFIDF_query(query, ngrams)
  cosine_sim = sparse_cosine_sim(tf_idf_query, tf_idf)
  cosine_sim = sorted(cosine_sim.items(), key=lambda x: x[1], reverse=True)
  print("Total ", len(cosine_sim), " documents found.\n")
  #print("Total ", len(cosine_sim), " documents found.\nTop ", min(k, len(cosine_sim)), " results are: ")

  l = []
  for i in cosine_sim:
      l.append(i[0])
  print(l)
  return l


def translate_hi2en(query):
  translator = google_translator()
  translated_text = translator.translate(query, lang_src='hi', lang_tgt='en')    
  return translated_text


query = "विश्व और राष्ट्रमंडल चैंपियन किम कोलिन्स "
translated_query = translate_hi2en(query)




Q = matching_score(10, translated_query,ngrams=2)
print_doc(Q[0])

Q = get_cos_sim_klist(10, translated_query, ngrams=2)
print_doc(Q[0])



#Evaluation
#10 queries were selected. From each domain 2 queries :- indomain and outdomain
"""
Politics: 
    Indomain:
              बुजुर्गों की मुफ्त देखभाल
              free care for the elderly
    Outdomain:
              लेबर पार्टी चुनाव अभियान
              labour party election campaign

Tech:
    Indomain:
              माइक्रोसॉफ्ट विंडोज एक्स पी के लिए अपग्रेड जारी करता है
              Microsoft releases upgrades for Windows XP

    Outdomain:
              गोपनीयता और दुरुपयोग के बारे में डर
              Fear about privacy and abuse

Entertainment:
    Indomain:
              award winning movies of this year
              इस वर्ष की पुरस्कार विजेता फिल्में

    Outdomain:
              nomination in the best actress category
             सर्वश्रेष्ठ अभिनेत्री श्रेणी में नामांकन

Sports:
    Indomain:
              विश्व और राष्ट्रमंडल चैंपियन किम कोलिन्स
              World and Commonwealth champion Kim Collins

    Outdomain:
              मैनचेस्टर यूनाइटेड प्लेयर रोनाल्डो
              Manchester United player Ronaldo
Business:
    Indomain:
              कच्चे तेल की बढ़ती कीमतें
              Rising crude oil prices

    Outdomain:
              देश की आर्थिक वृद्धि
              Economic growth of the country
"""

#the relevant list of documents found manually
#number indicate the filename in the folder
rel_tech1=[292,177,237,60,7]
rel_tech2=[31,26,61,79,86,164,227,321]
rel_politics1=[416,417]
rel_politics2=[99,107,249,251,254,260,264,268,305,307,316,318,333,344,346,394,405,413,414,415]
rel_business1=[221, 152, 138, 182, 30, 28, 381, 294, 110, 12, 45, 429, 273]
rel_business2=[493, 286, 505]
rel_sports1=[10, 92]
rel_sports2=[98]
rel_entertain1=[4,6,30,34,39,62,67,75,86,241,295,314,325,331,332,355,363,370,371]
rel_entertain2=[5,6,8,12,34,38,45,47,51,56,62,63,65,69,75,84,86,89,91,185,275,309,323,324,352,354,355,370]

rel_lst=[]
rel_lst.append(rel_tech1)
rel_lst.append(rel_tech2)
rel_lst.append(rel_politics1)
rel_lst.append(rel_politics2)
rel_lst.append(rel_business1)
rel_lst.append(rel_business2)
rel_lst.append(rel_sports1)
rel_lst.append(rel_sports2)
rel_lst.append(rel_entertain1)
rel_lst.append(rel_entertain2)

#query
q=[
"बुजुर्गों की मुफ्त देखभाल",
"लेबर पार्टी चुनाव अभियान",
"माइक्रोसॉफ्ट विंडोज एक्स पी के लिए अपग्रेड जारी करता है",
"गोपनीयता और दुरुपयोग के बारे में डर",
"इस वर्ष की पुरस्कार विजेता फिल्में",
"सर्वश्रेष्ठ अभिनेत्री श्रेणी में नामांकन",
"विश्व और राष्ट्रमंडल चैंपियन किम कोलिन्स",
"मैनचेस्टर यूनाइटेड प्लेयर रोनाल्डो",
"कच्चे तेल की बढ़ती कीमतें",
"देश की आर्थिक वृद्धि"
]


query=[]
for i in q:
  query.append(translate_hi2en(i))

"""
query=["Microsoft releases upgrades for Windows XP",
       "Fear about privacy and abuse",
       "free care for the elderly",
       "labour party election campaign",
       "Rising Crude oil prices",
       "China's economy growth",
       "Commonwealth Champion Kim Collins",
       "Manchester United player Cristiano Ronaldo",
       "This year's award-winning films",
       "Nomination in Best Actress Category"
       ]
"""



#Obtain the file name of the document by algorithm (cosine similarity)
l=[]
ret_lst={}

for j in range(0,10):
  l=[]
  print(query[j])
  Q = get_cos_sim_klist(2225, query[j], ngrams=3)
  if(Q==None):
    continue
  if(len(Q)==0):
    ret_lst[query[j]] = l
    continue
  for i in Q:
    st=dataset[i][0];
    number=int(st[st.find('/',len(st)-9)+1:st.find('.',len(st)-9)])
    l.append(number)
    
    q=query[j]
    ret_lst[q] = l

  print("\n"+ "Number of documents: "+ str(len(Q)))







#Average
def mean(lst):
  sum=0
  for i in lst:
    sum=sum+i
  sum=sum/len(lst)
  mean=int(sum*1000000)/1000000
  return mean


#Finds the intersection of 2 lists 
inter=[]
l=[]
for j in range(0,10):
  l=[]
  for i in rel_lst[j]:
    if i in ret_lst[query[j]]:
      l.append(i)
  inter.append(l)



#calcultes precision
#note 500 is the vaule given when it tries to divide by 0
print("PRECISION")

precision=[]
for i in range(0,10):
  if(len(ret_lst[query[i]]) ==0):
    precision.append(500)
    continue
  precision.append(len(inter[i])/len(ret_lst[query[i]]))
for i in precision:
  print(i)

print("\n")
print(mean(precision))


#calculates recall
#note 500 is the vaule given when it tries to divide by 0
print("\n")
print("RECALL")

recall=[]
for i in range(0,10):
  if(len(rel_lst[i]) ==0):
    recall.append(500)
  else:
    recall.append(len(inter[i])/len(rel_lst[i]))
for i in recall:
  print(i)
print("\n")
print(mean(recall))

print("\n\n")



#calculates fscore 
#note 500 is the vaule given when it tries to divide by 0
fscore=[]
for i in range(0,10):
  if((precision[i] + recall[i])==0):
    fscore.append(500)
  else:
    fscore.append((2*precision[i] * recall[i])/(precision[i] + recall[i]))

for i in fscore:
  print(i)
print("\n")
print(mean(fscore))
