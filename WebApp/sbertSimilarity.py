from sentence_transformers import SentenceTransformer
from sentence_transformers import models, losses
from nltk import sent_tokenize
import pandas as pd
import spacy
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import os
import fitz
import re
import docx2txt
import string

jd_RA=""
jd_LI=""
cvs= []
def process_bert_similarity():
            # This will download and load the pretrained model offered by SBERT.
            model = SentenceTransformer('all-mpnet-base-v2')

            # Although it is not explicitly stated in the official document of sentence transformer, the original BERT is meant for a shorter sentence. We will feed the model by sentences instead of the whole documents.
        
            sentences = sent_tokenize(jd_LI)
            base_embeddings_sentences = model.encode(sentences)
            base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)

            vectors = []
            for i, document in enumerate(cvs):
                    sentences = sent_tokenize(document)
                    embeddings_sentences = model.encode(sentences)
                    embeddings = np.mean(np.array(embeddings_sentences), axis=0)
                    vectors.append(embeddings)
                    print("making vector at index:", i)

            scores = cosine_similarity([base_embeddings], vectors).flatten()
            highest_score = 0
            highest_score_index = 0
            print(scores)
            for i, score in enumerate(scores):
                    if highest_score < score:
                        highest_score = score
                        highest_score_index = i

            most_similar_document = cvs[highest_score_index]
            print("Most similar document by BERT with the score =>",'resume = ',' | ',most_similar_document,' | ',' score =',highest_score ,' at index = ',highest_score_index)




# Define english stopwords
stop_words = stopwords.words('english')

# load the spacy module and create a nlp object
# This need the spacy en module to be present on the system.
nlp = spacy.load('en_core_web_sm')
# proces to remove stopwords form a file, takes an optional_word list
# for the words that are not present in the stop words but the user wants them deleted.


def remove_stopwords(text, stopwords=stop_words, optional_params=False, optional_words=[]):
    if optional_params:
        stopwords.append([a for a in optional_words])
    return [word for word in text if word not in stopwords]


def tokenize(text):
    # Removes any useless punctuations from the text
    text = re.sub(r'[^\w\s]', '', text)
    return word_tokenize(text)


def lemmatize(text):
    # the input to this function is a list
    str_text = nlp(" ".join(text))
    lemmatized_text = []
    for word in str_text:
        lemmatized_text.append(word.lemma_)
    return lemmatized_text

# internal fuction, useless right now.


def _to_string(List):
    # the input parameter must be a list
    string = " "
    return string.join(List)


def remove_tags(text, postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Takes in Tags which are allowed by the user and then elimnates the rest of the words
    based on their Part of Speech (POS) Tags.
    """
    filtered = []
    str_text = nlp(" ".join(text))
    for token in str_text:
        if token.pos_ in postags:
            filtered.append(token.text)
    return filtered

import spacy

try:
    nlp = spacy.load('en_core_web_sm')

except ImportError:
    print("Spacy's English Language Modules aren't present \n Install them by doing \n python -m spacy download en_core_web_sm")


def _base_clean(text):
    """
    Takes in text read by the parser file and then does the text cleaning.
    """
    text = tokenize(text)
    text = remove_stopwords(text)
    text = remove_tags(text)
    text = lemmatize(text)
    return text


def _reduce_redundancy(text):
    """
    Takes in text that has been cleaned by the _base_clean and uses set to reduce the repeating words
    giving only a single word that is needed.
    """
    return list(set(text))


def _get_target_words(text):
    """
    Takes in text and uses Spacy Tags on it, to extract the relevant Noun, Proper Noun words that contain words related to tech and JD. 
    """
    target = []
    sent = " ".join(text)
    doc = nlp(sent)
    for token in doc:
        if token.tag_ in ['NN', 'NNP']:
            target.append(token.text)
    return target


# https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
# https://towardsdatascience.com/the-best-document-similarity-algorithm-in-2020-a-beginners-guide-a01b9ef8cf05

def Cleaner(text):
    sentence =""
    sentence_cleaned = _base_clean(text)
    #sentence.append(sentence_cleaned)
    sentence_reduced = _reduce_redundancy(sentence_cleaned)
    #sentence.append(sentence_reduced)
    sentence_targetted = _get_target_words(sentence_reduced)
    sentence=sentence_cleaned
    #print(sentence)
    return sentence

import os
#import fitz
import docx2txt
import pandas as pd
def getresdata(file,data):
    text=""
    if file.endswith('.pdf'):
        for page in fitz.open('cvs txt/'+file):
            text = text + str(page.getText())
        text = " ".join(text.split('\n'))
        cvs.append(Cleaner(text))
    elif file.endswith('.docx'):
        temp = docx2txt.process('cvs txt/'+file)
        text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
        text = ' '.join(text)
        cvs.append(Cleaner(text))
    else:
        data=Cleaner(data)
        cvs.append(data)

# def listToString(s): 
#     print(type(s))
#     str1 = "" 
#     for ele in s: 
#           for a in ele:
#               str1 += a  
#               str1+=' '
#     return str1  

#cvs txt\ADNAN AHMED - CV for Research Assistants - Lab Instructors.txt
def func():
    in_dir='cvs txt/'
    data_paths = [i for i in (os.path.join(in_dir, f) for f in os.listdir(in_dir)) if os.path.isfile(i)]
    asst_instr=[]
    asst=[]
    instr=[]
    engr=[]
    jd=[]
    for path in data_paths:
        if 'Assistant' in path and 'Instructor' in path:
            asst_instr.append(path)
        elif 'Assistant' in path and 'Instructor' not in path:
            asst.append(path)
        elif 'Instructor'  in path and 'Assistant' not in path:
            instr.append(path)
        elif 'Engineer'  in path:
            engr.append(path)
        # elif 'JD/content/JD-Instructors.txt' or '/content/JD Research Assistants.txt' in path:
        #     jd.append(path)
    jd.append('WebApp/static/jd/JD Research Assistants.txt')
    jd.append('WebApp/static/jd/JD-Instructors.txt')

    with open(jd[0],'r',encoding='utf-8',errors='ignore') as f:
        jd_RA=f.read()
        jd_RA=Cleaner(jd_RA)
        
    with open(jd[1],'r',encoding='utf-8',errors='ignore') as f:
        jd_LI=f.read()
        jd_LI=Cleaner(jd_LI)
        jd_LI=_to_string(jd_LI)


    print('**************************')
    #print(jd_LI)
    print('**************************')
    for path in instr:
        with open(path,'r',encoding='utf-8',errors='ignore') as f:
            data=f.readlines()
            data=_to_string(data)
            data=Cleaner(data)
            data=_to_string(data)
            print(data)
            cvs.append(data)
        
            
    print('***********************')
    print(cvs)
    print('***********************')
process_bert_similarity()
print('***********************')
