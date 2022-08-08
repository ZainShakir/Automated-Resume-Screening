from flask import Blueprint,render_template, request, flash, redirect, url_for,current_app
from .models import Recruiter,User
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from . import db,ALLOWED_EXTENSIONS
from flask_login import login_user, login_required, logout_user, current_user
import os
from WebApp.screenResume import get_title
import pickle
import time
import pickle

auth=Blueprint('auth',__name__)
#resumeExtractor.pkl


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@auth.route('/recruiterLogin', methods=['GET', 'POST'])
def recruiter_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = Recruiter.query.filter_by(email=email).first()
        if user:
            if check_password_hash(user.password, password):
                flash('Logged in successfully!', category='success')
                login_user(user, remember=True)
                return redirect(url_for('views.home'))
            else:
                flash('Incorrect password, try again.', category='error')
        else:
            flash('Email does not exist.', category='error')
    return render_template('recruiterLogin.html', user=current_user )

@auth.route('/', methods=['GET','POST'])
def user_login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
       
        user = User.query.filter_by(email=email).first()
        if user:
                if check_password_hash(user.password, password):
                        flash('Logged in successfully!', category='success')
                        user.authenticated = True
                        # current_user.is_authenticated=True
                        db.session.add(user)
                        db.session.commit()
                        login_user(user, remember=True)
                        return redirect(url_for('views.jobList2'))
                else:
                        flash('Incorrect password, try again.', category='error')
        else:
                    flash('Email does not exist.', category='error')
            

    return render_template('applicantLogin.html', user=current_user)

'''
@auth.route('/applicantSignup', methods=['GET', 'POST'])
def applicant_signup():
    if request.method == 'POST':
        email = request.form.get('email')
        print(email)
        first_name = request.form.get('firstName')
        print(first_name)
        password1 = request.form.get('password1')
        print(password1)
        password2 = request.form.get('password2')
        print(password2)
        contact=request.form.get('contact')
        print(contact)
        if 'resume' not in request.files:
            print('helllo')
            flash('No file Found',category='error')
            #return redirect(request.url)
        resume=request.files['resume']
        
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        else:
            if len(email) < 4:
                flash('Email must be greater than 3 characters.', category='error')
            elif len(first_name) < 2:
                flash('First name must be greater than 1 character.', category='error')
            elif password1 != password2:
                flash('Passwords don\'t match.', category='error')
            elif len(password1) < 5:
                flash('Password must be at least 5 characters.', category='error')
            elif len(contact) != 11:
                flash('Enter a valid contact#', category='error')
            elif resume.filename == '':
                flash('No File Selected',category='error')
                #return redirect(request.url)
            elif resume and not allowed_file(resume.filename):
                flash('Allowed File extensions : .txt , .pdf , .docx',category='error')
                #return redirect(request.url)
            else:
                filename = secure_filename(resume.filename)
                path=os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
              #  data=resumeExtractor.extractorData(path,resume.filename.rsplit('.',1)[1].lower())
                resume.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
                new_user = User(email=email, first_name=first_name, password=generate_password_hash(
                password1, method='sha256'),contact=contact,resume_file=resume.filename)
                db.session.add(new_user)
                new_user_resume=User_skills(user_id=2,skills=data[5])
                db.session.add(new_user_resume)
                db.session.commit()
                login_user(new_user, remember=True)
                flash('Account created!', category='success')
                return redirect(url_for('auth.user_login'))
        
    return render_template('applicantSignup.html',user=current_user)
'''

# @auth.route('/applicantSignup', methods=['GET', 'POST'])
# def applicant_signup():
#     if request.method == 'POST':
#         email = request.form.get('email')
#         print(email)
#         first_name = request.form.get('firstName')
#         print(first_name)
#         password1 = request.form.get('password1')
#         print(password1)
#         password2 = request.form.get('password2')
#         print(password2)
#         contact=request.form.get('contact')
#         print(contact)
#         if 'resume' not in request.files:
#             print('helllo')
#             flash('No file Found',category='error')
#             #return redirect(request.url)
#         resume=request.files['resume']
#         print(resume)
        
#         user = User.query.filter_by(email=email).first()
#         if user:
#             flash('Email already exists.', category='error')
#         else:
#             if len(email) < 4:
#                 flash('Email must be greater than 3 characters.', category='error')
#             elif len(first_name) < 2:
#                 flash('First name must be greater than 1 character.', category='error')
#             elif password1 != password2:
#                 flash('Passwords don\'t match.', category='error')
#             elif len(password1) < 5:
#                 flash('Password must be at least 5 characters.', category='error')
#             elif len(contact) != 11:
#                 flash('Enter a valid contact#', category='error')
#             elif resume.filename == '':
#                 flash('No File Selected',category='error')
#                 #return redirect(request.url)
#             elif resume and not allowed_file(resume.filename):
#                 flash('Allowed File extensions : .txt , .pdf , .docx',category='error')
#                 #return redirect(request.url)
#             else:
#                 filename = secure_filename(resume.filename)
#                 resume.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
#                 new_user = User(email=email, first_name=first_name, password=generate_password_hash(
#                 password1, method='sha256'),contact=contact,resume_file=resume.filename)
#                 db.session.add(new_user)
#                 db.session.commit()
#                 #login_user(new_user, remember=True)
#                 flash('Account created!', category='success')
#                 return redirect(url_for('auth.user_login'))
        
#     return render_template('applicantSignup.html')
@auth.route('/applicantSignup', methods=['GET', 'POST'])
def applicant_signup():
    if request.method == 'POST':
        email = request.form.get('email')
        print(email)
        first_name = request.form.get('firstName')
        print(first_name)
        password1 = request.form.get('password1')
        print(password1)
        password2 = request.form.get('password2')
        print(password2)
        contact=request.form.get('contact')
        print(contact)
        if 'resume' not in request.files:
            print('helllo')
            flash('No file Found',category='error')
            #return redirect(request.url)
        resume=request.files['resume']
        print(resume)
        
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        else:
            if len(email) < 4:
                flash('Email must be greater than 3 characters.', category='error')
            elif len(first_name) < 2:
                flash('First name must be greater than 1 character.', category='error')
            elif password1 != password2:
                flash('Passwords don\'t match.', category='error')
            elif len(password1) < 5:
                flash('Password must be at least 5 characters.', category='error')
            elif len(contact) != 11:
                flash('Enter a valid contact#', category='error')
            elif resume.filename == '':
                flash('No File Selected',category='error')
                #return redirect(request.url)
            elif resume and not allowed_file(resume.filename):
                flash('Allowed File extensions : .txt , .pdf , .docx',category='error')
                #return redirect(request.url)
            else:
                filename = secure_filename(resume.filename)
                
                resume.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
                
                time.sleep(5)

                with open('WebApp/static/resume/'+filename,'r',encoding='utf-8',errors='ignore') as f1:
                  sentence2=f1.read()
                output=get_title(sentence2)
                skill=[]
                for keys in output.keys():
                    skill.append(keys)
                    
                new_user = User(email=email, first_name=first_name, password=generate_password_hash(
                password1, method='sha256'),contact=contact,resume_file=resume.filename,skill1=skill[0],skill2=skill[1])
                db.session.add(new_user)
                db.session.commit()
                #login_user(new_user, remember=True)
                flash('Account created!', category='success')
                return redirect(url_for('auth.user_login'))
        
    return render_template('applicantSignup.html')


'''

@auth.route('/logout',methods=['GET'])
@login_required
def logout():
    user = current_user
    user.authenticated = False
    db.session.add(user)
    db.session.commit()
    logout_user()
    return redirect(url_for('auth.user_login'))

'''

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.user_login'))
