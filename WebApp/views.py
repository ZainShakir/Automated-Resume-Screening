from concurrent.futures import process
from sre_constants import SUCCESS
from flask import Blueprint,render_template, request, flash, jsonify,current_app, redirect, url_for
from flask_login import login_required, current_user
from .models import Job,Applied_job,User
from . import db,ALLOWED_EXTENSIONS2
from werkzeug.utils import secure_filename
import os
import json
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
from WebApp.res import resumeExtractor
import fitz
import re
import docx2txt
import string

views=Blueprint('views',__name__)
cvs=[]
rows, cols = (5, 5)
arr = [[0 for i in range(cols)] for j in range(rows)]
rows2, cols2 = (5, 5)
temp = [[0 for i in range(cols)] for j in range(rows)]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS2

@views.route('/home',methods=['GET'])
def home():
     
     return render_template("listJob.html", user=current_user)

@views.route('/home2',methods=['GET', 'POST'])
@login_required
def home2():
     return render_template("viewjobs.html", user=current_user)
@views.route('/delete-job', methods=['GET', 'POST'])
def delete_job():
     job = json.loads(request.data)
     jobId = job['jobId']
     job = Job.query.get(jobId)
     if job:
          if job.recruiter_id == current_user.id:
               db.session.delete(job)
               db.session.commit()
     return jsonify({})

@views.route('/job-listing',methods=['GET', 'POST'])
def jobList():
     return render_template('listJob.html',user=current_user)
@views.route('/job-listing2',methods=['GET', 'POST'])
@login_required
def jobList2():
     if request.method == 'POST':
          empid=current_user.id
          job_id=request.form.get('jobid')
          rec_id=request.form.get('recid')
          new_applicant=Applied_job(job_id=job_id,rec_id=rec_id,emp_id=empid)
          db.session.add(new_applicant)
          db.session.commit()
          flash('Successfully Applied!', category='success')
     return render_template('viewjobs.html',user=current_user)
 
@views.route('/apply',methods=['GET', 'POST'])
def apply():
     # if request.method == 'POST':
     #      empid=current_user.id
     #      new_applicant=Applied_job(job_id=2,emp_id=empid)
     #      db.session.add(new_applicant)
     #      db.session.commit()
     #      flash('Successfully Applied!', category='success')
     return render_template('applyToJob.html',user=current_user)

@views.route('/post-job',methods=['GET','POST'])
# @login_required
def jobpost():
          if request.method == 'POST':
               position = request.form.get('position')
               mode = request.form.get('mode')
               salary = request.form.get('salary')
               job_description=str(position) +'.'+str(mode) +'.'+'PKR '+str(salary)
               if 'jd' not in request.files:
                    flash('No file part',category='error')
               jd=request.files['jd']

               if len(job_description) < 3:
                    flash('Job Description is too short!', category='error')
               elif jd.filename == '':
                    flash('No File Selected',category='error')
               elif jd and not allowed_file(jd.filename):
                    flash('Allowed File extensions : .txt , .pdf , .docx',category='error')
               else:
                    filename = secure_filename(jd.filename)
                    jd.save(os.path.join(current_app.config['UPLOAD_FOLDER2'], filename))
                    new_job = Job(jd_filename=jd.filename,description=job_description, recruiter_id=current_user.id)
                    db.session.add(new_job)
                    db.session.commit()
                    flash('Job added!', category='success')
                
          return render_template('post_job.html',user=current_user)

def process_bert_similarity(jd):
               # This will download and load the pretrained model offered by SBERT.
               model = SentenceTransformer('all-mpnet-base-v2')
               sentences = sent_tokenize(jd)
               base_embeddings_sentences = model.encode(sentences)
               base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)

               vectors = []
               for i, document in enumerate(cvs):
                         sentences = sent_tokenize(document)
                         #print(sentences)
                         embeddings_sentences = model.encode(sentences)
                         embeddings = np.mean(np.array(embeddings_sentences), axis=0)
                         #print(embeddings)
                         vectors.append(embeddings)
                         print("making vector at index:", i)

               scores = cosine_similarity([base_embeddings], vectors).flatten()
               # highest_score = 0
               # highest_score_index = 0
               # #print(scores)
               # for i, score in enumerate(scores):
               #           if highest_score < score:
               #                highest_score = score
               #                highest_score_index = i

               # most_similar_document = cvs[highest_score_index]
               # print("Most similar document by BERT with the score =>",'resume = ',' | ',most_similar_document,' | ',' score =',highest_score ,' at index = ',highest_score_index)
               return scores
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

@views.route('/view-applicants',methods=['GET','POST'])
def view_applicants():
     if request.method == 'POST':
          jid=request.form.get('jid')
          job=Job.query.filter_by(id=jid).first()
          
          if job:
               #WebApp\static\jd
               with open('WebApp/static/jd/'+job.jd_filename,'r',encoding='utf-8',errors='ignore') as f:
                              jd=f.read()
                              jd=Cleaner(jd)
                              jd=_to_string(jd)
               applicants=Applied_job.query.filter_by(job_id=job.id).all()
               for i in range(len(applicants)):
                    applicant=User.query.filter_by(id=applicants[i].emp_id).first()
                    path='WebApp/static/cvs/'
                    #flash(applicant.first_name,category='success')
                    #WebApp\static\cvs\Ebad Ali - CV for Research Assistant.txt
                    text=resumeExtractor.extractorData(path+applicant.resume_file,'txt')
                    data=_to_string(text[3])
                    cvs.append(data)
                    score=0
                    f.close()
                    print('************************')
                    arr[i][0]=applicant.first_name
                    arr[i][1]=text[2]
                    arr[i][2]=text[3]
                    arr[i][3]=score
                    arr[i][4]=applicant.resume_file
                    #print(arr)
                    
               scores=process_bert_similarity(jd)
               print(scores)
               for i in range(len(scores)):
                    if arr[i]:
                         arr[i][3]=scores[i]
                    else:
                         break
               for i in range(len(scores),len(arr)):
                    for j in range(5):
                         arr[i][j]=0
               
               #arr = [[0 for i in range(cols)] for j in range(rows)]
               print(arr)
               cvs.clear()
               flash('Showing Results for '+job.description.split('.')[0],category='Success')
          else:
               flash('not found',category='error')
     return render_template('viewapplicants.html',data=arr,len=len(arr))
     


@views.route('/search-candidate',methods=['GET', 'POST'])
def search():
     data=[]
     title=""
     if request.method=='POST':
         title=request.form.get('catem')
         candidates=User.query.filter_by(skill1=title).all()
         candidates1=User.query.filter_by(skill2=title).all()
         if candidates or candidates1:  
             for temp in candidates:
                 data.append(temp)
             for temp1 in candidates1:
                 data.append(temp1)
             flash("Showing results for " + title ,category='success')
         else:
            flash('No Candidates found',category='error')
            
     return render_template("searchCandidate.html",user=current_user, checkdata=data,title=title)