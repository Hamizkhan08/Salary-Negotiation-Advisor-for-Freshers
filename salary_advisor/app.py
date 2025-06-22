from flask import Flask, render_template, request
import pickle
import numpy as np
import os
from werkzeug.utils import secure_filename
import PyPDF2
import spacy
import re  # Import the regular expressions library

# --- INITIALIZE NLP MODEL ---
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load ML model
model = pickle.load(open("model/salary_model.pkl", "rb"))
stream_encoder = pickle.load(open("model/stream_encoder.pkl", "rb"))
stream_labels = list(stream_encoder.classes_)

@app.route('/')
def home():
    return render_template('index.html', streams=stream_labels)

@app.route('/predict', methods=["POST"])
def predict():
    try:
        gender = int(request.form['gender'])
        tenth = float(request.form['tenth'])
        twelfth = float(request.form['twelfth'])
        cgpa = float(request.form['cgpa'])
        communication = int(request.form['communication'])
        internship = int(request.form['internship'])
        training = int(request.form['training'])
        technical = int(request.form['technical'])
        stream_name = request.form['stream']
        stream = stream_encoder.transform([stream_name])[0]
        features = np.array([[gender, tenth, twelfth, cgpa, communication,
                              internship, training, technical, stream]])
        prediction = model.predict(features)[0]
        salary_lpa = round(prediction / 100000, 2)
        return render_template('result.html', salary=salary_lpa)
    except Exception as e:
        return render_template('result.html', error=f"Prediction failed: {e}")


# --- FINAL, MOST ROBUST AI EXTRACTION ENGINE ---
def extract_salary_from_text(text):
    """
    A more robust function to find salary figures in text.
    It uses multiple methods: spaCy entities and several regular expressions.
    """
    # Normalize text: remove spaces around commas and newlines
    text = re.sub(r'\s*,\s*', ',', text)
    text = text.replace('\n', ' ')

    # Method 1: Use spaCy's Named Entity Recognition (Best for clean text)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            salary_text = ent.text.replace('$', '').replace(',', '').replace('₹', '').strip()
            try:
                if 'lakh' in ent.text.lower() or 'lacs' in ent.text.lower():
                    number_part = re.search(r'(\d+\.?\d*)', salary_text)
                    if number_part:
                        return float(number_part.group(1)) * 100000
                return float(salary_text)
            except (ValueError, TypeError):
                continue

    # Method 2: Regex for Indian formats (e.g., "Rs. 4,50,000", "4.5 Lakhs")
    pattern_lakhs = re.compile(r'(\d+\.?\d*)\s*(?:lakh|lacs)')
    pattern_full = re.compile(r'(?:Rs|INR|₹)\.?\s*([\d,]+(?:,\d+)*)')
    
    match = pattern_lakhs.search(text)
    if match:
        return float(match.group(1)) * 100000
    
    match = pattern_full.search(text)
    if match:
        return float(match.group(1).replace(',', ''))

    # Method 3: "Last Resort" Regex - Find any 6 or 7-digit number near salary keywords
    # This is less precise but catches many tricky cases.
    salary_keywords = ['ctc', 'salary', 'compensation', 'remuneration', 'annum']
    for keyword in salary_keywords:
        # Find the keyword, then look for a number in the next 50 characters
        keyword_match = re.search(keyword, text, re.IGNORECASE)
        if keyword_match:
            search_area = text[keyword_match.end():keyword_match.end()+50]
            # Look for a number with 6 or 7 digits, possibly with commas
            number_match = re.search(r'([\d,]{6,8})', search_area)
            if number_match:
                try:
                    return float(number_match.group(1).replace(',', ''))
                except (ValueError, TypeError):
                    continue
    
    return None


@app.route('/analyze_offer', methods=["POST"])
def analyze_offer():
    try:
        if 'offer_letter_file' not in request.files:
            return render_template('result.html', error="No file part in the form.")
        file = request.files['offer_letter_file']
        if file.filename == '':
            return render_template('result.html', error="No file selected for uploading.")
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 1. Extract text from the PDF
            pdf_text = ""
            with open(filepath, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    pdf_text += page.extract_text()
            
            # --- DEBUGGING PRINT ---
            print("----------- EXTRACTED PDF TEXT -----------")
            print(pdf_text)
            print("------------------------------------------")

            # 2. Use our smarter function to get the salary
            offered_salary = extract_salary_from_text(pdf_text)
            
            # 3. Generate a review
            if offered_salary is None or offered_salary == 0:
                review_text = "AI Analysis: Could not automatically determine the salary from the document. This can happen with complex table formats. Please ensure the salary is clearly mentioned (e.g., ₹4,50,000 or 4.5 Lakhs)."
            else:
                # Simple scoring logic
                score = 0
                baseline_salary = 450000
                if offered_salary > baseline_salary * 1.1: score += 3
                elif offered_salary >= baseline_salary * 0.9: score += 2
                else: score += 1
                
                if score >= 3:
                    review_text = f"AI Analysis: Excellent Offer! The detected salary of ₹{offered_salary:,.0f} per year is highly competitive and appears to be above the industry standard for freshers."
                elif score == 2:
                    review_text = f"AI Analysis: Solid Offer. The detected salary of ₹{offered_salary:,.0f} per year is competitive and aligns well with industry standards."
                else:
                    review_text = f"AI Analysis: Room for Improvement. The detected salary of ₹{offered_salary:,.0f} per year seems slightly below the industry standard. You may have room to negotiate."

            return render_template('result.html', offer_analysis=review_text)

        else:
            return render_template('result.html', error="Invalid file type. Please upload a PDF.")

    except Exception as e:
        return render_template('result.html', error=f"An unexpected error occurred during analysis: {e}")

if __name__ == "__main__":
    app.run(debug=True)