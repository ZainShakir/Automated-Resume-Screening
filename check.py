import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
import re
import json
import pickle
import os
import sys,fitz

def clean_text(resume_text):
    stopwords_set = set(stopwords.words('english')+['``',"''"])
    resume_text = re.sub('http\S+\s*', ' ', resume_text)  # remove URLs
    resume_text = re.sub('RT|cc', ' ', resume_text)  # remove RT and cc
    resume_text = re.sub('#\S+', '', resume_text)  # remove hashtags
    resume_text = re.sub('@\S+', '  ', resume_text)  # remove mentions
    resume_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resume_text)  # remove punctuations
    resume_text = re.sub(r'[^\x00-\x7f]',r' ', resume_text) 
    resume_text = re.sub('\s+', ' ', resume_text)  # remove extra whitespace
    resume_text = resume_text.lower()  # convert to lowercase
    resume_text_tokens = word_tokenize(resume_text)  # tokenize
    filtered_text = [w for w in resume_text_tokens if not w in stopwords_set]  # remove stopwords
    return ' '.join(filtered_text)


def get_title(text):
    with open('AI-IR/LSTM MODEL/feature_tokenizer.pickle', 'rb') as handle:
        feature_tokenizer = pickle.load(handle)         
            
    with open('AI-IR/LSTM MODEL/dictionary.pickle', 'rb') as handle:
        encoding_to_label = pickle.load(handle)

    with open("AI-IR/LSTM MODEL/labels.json", "r") as read_file:
                original_labels = json.load(read_file)
    
    

    sen=clean_text(text)
    text=sen
    

    max_length = 500
    trunc_type = 'post'
    padding_type = 'post'

    predict_sequences = feature_tokenizer.texts_to_sequences([text])
    predict_padded = pad_sequences(predict_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    predict_padded = np.array(predict_padded)

    model = keras.models.load_model('AI-IR/LSTM MODEL/LSTM_model')
    prediction = model.predict(predict_padded)

    encodings = np.argpartition(prediction[0], -8)[-8:]
    encodings = encodings[np.argsort(prediction[0][encodings])]
    encodings = reversed(encodings)

    data = {}

    for encoding in encodings:
        label = encoding_to_label[encoding]
        probability = prediction[0][encoding] * 100
        probability = round(probability, 2)
        data[original_labels[label]]=probability
    print(data.keys())
    if 'Civil Engineer' in data.keys():
            del(data['Civil Engineer'])
    print('******************************')
    print(data)

    titles=[[],[]]
    for key,values in data.items():
        titles[0].append(key)
        titles[1].append(values)
    
    new_data={}
    new_data[titles[0][0]]=titles[1][0]
    new_data[titles[0][1]]=titles[1][1]

    return new_data

# with open("AI-IR/WebApp/static/resume/ADNAN_AHMED_-_CV_for_Research_Assistants_-_Lab_Instructors.txt",'r',encoding='utf-8',errors='ignore') as f1:
#     sentence2=f1.read()
#/Zohaib Khan - CV for Research Assistant - Lab Instructor.txt
text=""
#AI-IR\WebApp\static\resume\Ali_Rehmans_Resume_.pdf
path='''AI-IR/WebApp/static/resume/Ali_Rehmans_Resume_.pdf'''
file=path.split('/')
print(file)
for page in fitz.open('AI-IR/WebApp/static/resume/'+file[4]):
  text = text + str(page.getText())
  text = " ".join(text.split('\n'))
test=get_title(text)
print(test)




