# 💼 Salary Negotiation Advisor for Freshers

An intelligent web application that predicts a fresher's expected salary based on academic and personal profile, and analyzes uploaded offer letters to provide AI-generated negotiation advice.
![image](https://github.com/user-attachments/assets/58a5eac2-16df-4afb-a31a-ecef8416b84f)

![image](https://github.com/user-attachments/assets/1ac16369-0b32-47d7-89bb-c13c9919533c)

![Uploading image.png…]()

## 🧠 Features

* 📊 **Salary Prediction** based on:

  * Gender
  * 10th & 12th marks
  * CGPA
  * Communication level
  * Internship, training & technical course experience
  * Academic stream
* 📄 **Offer Letter Analysis**: Upload a PDF offer letter and receive insights:

  * Extracted salary
  * Review: Excellent, Solid, or Room for Improvement
  * Negotiation recommendations

---

## 📁 Project Structure

```
salary-negotiation-advisor/
│
├── app.py                   # Flask web application
├── train_model.py          # Model training and preprocessing script
├── dataset/
│   └── Sample_with_Salary.csv
├── model/
│   ├── salary_model.pkl     # Trained salary prediction model
│   └── stream_encoder.pkl   # LabelEncoder for academic streams
├── uploads/                # Directory for uploaded offer PDFs
├── templates/
│   ├── index.html           # Input form for prediction
│   └── result.html          # Displays results & analysis
└── requirements.txt         # Python dependencies
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/salary-negotiation-advisor.git
cd salary-negotiation-advisor
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Ensure you have `spaCy` and the English model installed:

```bash
python -m spacy download en_core_web_sm
```

### 4. Train the Model (Optional: if retraining is needed)

```bash
python train_model.py
```

### 5. Run the Flask App

```bash
python app.py
```

App runs at `http://127.0.0.1:5000`

---

## 🖥️ Usage

### 🔹 Predict Salary

1. Fill in academic and personal details on the home page.
2. Submit to get the estimated salary in LPA.

### 🔹 Analyze Offer Letter

1. Upload a `.pdf` offer letter file.
2. The app extracts the salary and gives feedback on how good the offer is for a fresher.

---

## 📦 Dependencies

* Flask
* NumPy
* Pandas
* Scikit-learn
* spaCy
* PyPDF2
* Werkzeug

(Add these in `requirements.txt` if not already.)

---

## 📌 Future Improvements

* Add login/user dashboard
* Expand dataset for better prediction
* Improve offer parsing accuracy with NLP transformers

---

## 📄 License

MIT License

---
