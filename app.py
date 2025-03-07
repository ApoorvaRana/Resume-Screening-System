from flask import Flask, request, render_template, send_from_directory
import os
import pdfplumber
from docx import Document
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# MySQL database configuration (XAMPP)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/resume_matcher'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Add this line to enable migrations

# Ensure upload folder exists
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Create folder if not exists

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return "No file uploaded", 400

    file = request.files['resume']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)  # Save the file

    return f"File {file.filename} uploaded successfully!"

# Database Model
# class Resume(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     filename = db.Column(db.String(200), nullable=False)
#     similarity_score = db.Column(db.Float, nullable=False)

class Resume(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False, unique=True)  # UNIQUE constraint
    similarity_score = db.Column(db.Float, nullable=False)

# Create tables in MySQL
with app.app_context():
    db.create_all()

# Load NLP model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Text preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    words = [stemmer.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)

# Resume text extraction functions
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        return extract_text_from_txt(file_path)
    return ""

@app.route("/")
def matchresume():
    return render_template('matchresume.html')

@app.route('/matcher', methods=['POST'])
def matcher():
    job_description = request.form['job_description']
    resume_files = request.files.getlist('resumes')

    resumes = []
    filenames = []
    
    for resume_file in resume_files:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(filename)
        text = extract_text(filename)
        resumes.append(preprocess_text(text))
        filenames.append(resume_file.filename)

    if not resumes or not job_description:
        return render_template('matchresume.html', message="Please upload resumes and enter a job description.")

    # Compute embeddings
    job_embedding = model.encode([preprocess_text(job_description)])[0]
    resume_embeddings = np.array([model.encode([resume])[0] for resume in resumes])
    similarities = cosine_similarity([job_embedding], resume_embeddings)[0]

    # Get top matches
    top_indices = similarities.argsort()[-5:][::-1]
    top_resumes = [filenames[i] for i in top_indices]
    similarity_scores = [round(similarities[i], 2) for i in top_indices]

    # Store results in MySQL
    # for i in range(len(top_resumes)):
    #     db.session.add(Resume(filename=top_resumes[i], similarity_score=similarity_scores[i]))
    # db.session.commit()

    for i in range(len(top_resumes)):
        existing_resume = Resume.query.filter_by(filename=top_resumes[i]).first()
        if not existing_resume:  # Only add if not already present
            db.session.add(Resume(filename=top_resumes[i], similarity_score=similarity_scores[i]))
    db.session.commit()


    return render_template('matchresume.html', message="Top matching resumes:", top_resumes=top_resumes, similarity_scores=similarity_scores)

# Route to view stored resumes in MySQL
# @app.route('/view_resumes')
# def view_resumes():
#     resumes = Resume.query.order_by(Resume.similarity_score.desc()).all()
#     return render_template('view_resumes.html', resumes=resumes)

# @app.route('/view_resumes')
# def view_resumes():
#     resumes = []
#     for filename in os.listdir(UPLOAD_FOLDER):
#         resumes.append({"filename": filename, "similarity_score": "N/A"})  # Replace with actual scores if needed
    
#     return render_template('view_resumes.html', resumes=resumes)

@app.route('/view_resumes')
def view_resumes():
    resumes = Resume.query.order_by(Resume.similarity_score.desc()).all()
    
    # Use a set to filter unique filenames
    unique_resumes = {}
    for resume in resumes:
        if resume.filename not in unique_resumes:  # Avoid duplicates
            unique_resumes[resume.filename] = resume.similarity_score

    return render_template('view_resumes.html', resumes=[{"filename": k, "similarity_score": v} for k, v in unique_resumes.items()])


if __name__ == '__main__':
    app.run(debug=True)
