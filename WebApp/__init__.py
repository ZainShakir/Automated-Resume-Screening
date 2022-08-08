from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from os import path
from flask_login import LoginManager
from scipy.fft import idst
from werkzeug.utils import secure_filename
import pickle
#from WebApp.res import resumeExtractor
import pandas as pd

db = SQLAlchemy()
DB_NAME = "database.db"
UPLOAD_FOLDER = 'WebApp/static/resume/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
#WebApp\static\resume\ADNAN_AHMED_-_CV_for_Research_Assistants_-_Lab_Instructors.txt
#D:\SEM 6\AIproj\AI-IR\WebApp\static\jd\JD_Research_Assistants.txt
UPLOAD_FOLDER2 = 'WebApp/static/jd/'
ALLOWED_EXTENSIONS2 = {'txt', 'pdf', 'docx'}
# extractorObj = pickle.load(open("resumeExtractor.pkl","rb"))
#D:\SEM 6\AI-IR (1)\AI-IR\WebApp\static\jd
def create_app():
    app=Flask(__name__)
    app.config['SECRET_KEY']='ali19k1279 ZAIN'
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_NAME}'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2
    #app.config['resumeExtractor']=resumeExtractor
    db.init_app(app)
    
    from .views import views
    from .auth import auth
    
    app.register_blueprint(views,url_prefix='/')
    app.register_blueprint(auth,url_prefix='/')
    
    from .models import Job,Recruiter,User

    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.user_login'
    login_manager.init_app(app)

    @login_manager.user_loader
    def load_user(id):
        check=User.query.get(int(id))
        print(check.email)
        return User.query.get(int(id))
    @login_manager.user_loader
    def load_recruiter(id):
        return Recruiter.query.get(int(id))

    return app

def create_database(app):
    if not path.exists('WebApp/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')