from flask import Flask, request, render_template, send_from_directory
import os
import docx2txt
import PyPDF2
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import pipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# -------- TEXT EXTRACTION -------- #

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

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
    else:
        return ""

# -------- CLEAN TEXT -------- #

def clean_text(text):
    text = text.lower()

    # remove emails & phone numbers
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\d{10,}', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# -------- ROUTES -------- #

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(filepath)

        text = extract_text(filepath)
        if text.strip():
            resumes.append(text)
            filenames.append(resume_file.filename)

    if not resumes or not job_description.strip():
        return render_template('matchresume.html',
                               message="Please upload valid resumes and enter a job description.")

    vectorizer = TfidfVectorizer().fit_transform([job_description] + resumes)
    vectors = vectorizer.toarray()

    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    similarities = cosine_similarity([job_vector], resume_vectors)[0]

    similarity_scores = [round(score * 100, 2) for score in similarities]

    results = list(zip(filenames, similarity_scores))
    results = sorted(results, key=lambda x: x[1], reverse=True)[:5]

    return render_template('matchresume.html',
                           message="Top Matching Resumes",
                           results=results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/interview', methods=['POST'])
def interview():
    resume_name = request.form['selected_resume']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], resume_name)

    resume_text = extract_text(filepath)

    # ✅ Clean + limit text
    resume_text = clean_text(resume_text)[:800]

    job_description = request.form.get('job_description', '')
    job_description = clean_text(job_description)[:500]

    questions = generate_questions_from_resume(resume_text, job_description)

    return render_template("interview.html",
                           resume_name=resume_name,
                           questions=questions)

# -------- LOAD MODEL -------- #

question_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# -------- QUESTION GENERATION -------- #
def extract_skills(text):
    skills = [
        "python", "java", "c++", "sql", "machine learning",
        "deep learning", "opencv", "flask", "django",
        "html", "css", "javascript", "react",
        "node", "aws", "cloud", "dbms"
    ]

    text = text.lower()
    found = [skill for skill in skills if skill in text]

    return found
def generate_questions_from_resume(resume_text, job_description):
    skills = extract_skills(resume_text + " " + job_description)

    if not skills:
        return ["No technical skills detected."]

    questions = []

    # ✅ Strong templates (forces company-style questions)
    templates = [
        "Explain how {skill} is used in real-world applications.",
        "How would you optimize performance when using {skill}?",
        "Describe a project where you used {skill} and challenges faced.",
        "How does {skill} handle scalability in large systems?",
        "What are common issues faced when working with {skill} and how do you solve them?",
        "How would you design a system using {skill}?"
    ]

    for i, skill in enumerate(skills[:6]):
        template = templates[i % len(templates)]

        prompt = f"""
Convert the following into a proper technical interview question:

{template.replace("{skill}", skill)}

Only return the question.
"""

        output = question_generator(
            prompt,
            max_length=60,
            do_sample=False   # 🔥 VERY IMPORTANT (no randomness)
        )[0]['generated_text']

        q = output.strip()

        # Ensure proper formatting
        if not q.endswith("?"):
            q += "?"

        questions.append(q)

    return questions[:5]

# -------- RUN -------- #

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)