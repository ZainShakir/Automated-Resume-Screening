from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    jd_filename=db.Column(db.String(100))
    description = db.Column(db.String(10000))
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    recruiter_id = db.Column(db.Integer, db.ForeignKey('recruiter.id'))

class Applied_job(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    job_id=db.Column(db.Integer,db.ForeignKey('job.id'))
    rec_id=db.Column(db.Integer,db.ForeignKey('recruiter.id'))
    emp_id=db.Column(db.Integer,db.ForeignKey('user.id'))
    
    
class Recruiter(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    contact=db.Column(db.String(11))
    job = db.relationship('Job')
    
class User_skills(db.Model):
    id=db.Column(db.Integer,primary_key=True)
    user_id=db.Column(db.Integer,db.ForeignKey('user.id'))
    skills=db.Column(db.String(100000))
    
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    contact=db.Column(db.String(10))
    skill1=db.Column(db.String(20))
    skill2=db.Column(db.String(20))
    resume_file=db.Column(db.String(30))

