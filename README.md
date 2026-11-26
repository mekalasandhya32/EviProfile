
# EviProfile
### An Intelligent Pipeline for Automated Candidate Profiling and Resume Intelligence Extraction

EviProfile is an AI/ML-powered resume intelligence system designed to automate resume analysis, skill extraction, candidate profiling, credibility assessment, and job-role matching. It combines machine learning, natural language processing, and rule-based techniques to generate structured insights from resumes.

## 🚀 Features

- **Resume Parsing:** Extracts text from PDF, DOCX, and TXT resumes.
- **Skill Extraction:** Automatically identifies technical and professional skills.
- **Skill Evidence Score (SES):** Analyzes skill-related evidence in candidate profiles.
- **Profile Credibility Index (PCI):** Evaluates profile credibility based on extracted information and evidence.
- **Skill Level Analysis:** Estimates skill proficiency using contextual evidence.
- **Skill Graph:** Represents relationships between extracted skills.
- **Job-Field Classification:** Predicts candidate job domains using machine learning.
- **Job Matching & Ranking:** Matches candidate profiles with job requirements and ranks candidates.
- **Resume Credibility Analysis:** Detects potentially inflated resume claims.
- **ML Model Benchmarking:** Compares multiple machine learning algorithms.
- **Web Application:** Interactive interface developed using Flask.

## 🧠 Machine Learning Approach

EviProfile uses a hybrid approach combining machine learning models with rule-based intelligence.

### Main Classification Model

- **Feature Extraction:** TF-IDF
- **Classifier:** Multi-Layer Perceptron (MLP)
- **Hidden Layers:** 256 and 64 neurons
- **Optimizer:** Adam
- **Training Strategy:** Early stopping
- **Classification:** 10 job fields

### Models Evaluated

- Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest
- Multi-Layer Perceptron (MLP)

## 🔄 System Workflow

1. Resume Upload/Input
2. Resume Text Extraction
3. Text Preprocessing
4. Skill and Profile Information Extraction
5. Skill Evidence Analysis
6. Profile Credibility Scoring
7. Job-Field Classification
8. Candidate Matching and Ranking
9. Structured Candidate Insights

## 📊 Dataset

The project uses a synthetic resume dataset for training and evaluation.

| Dataset Information | Value |
|---|---:|
| Total resumes | 2,500 |
| Training samples | 2,000 |
| Testing samples | 500 |

> The dataset is synthetic and intended for development and experimental evaluation. Results may differ on real-world resumes.

## 📈 Evaluation Results

Reported experimental results on the synthetic dataset:

| Metric | Result |
|---|---:|
| Skill Extraction F1 Score | 0.995 |
| Experience Extraction MAE | 0.20 years |
| Skill Level Spearman Correlation | 0.757 |
| Skill Evidence Spearman Correlation | 0.829 |
| Inflated Resume Detection AUC | 0.943 |
| Candidate Ranking nDCG@10 | 0.961 |
| Inflated Candidates in Top 10 | 0.8% |
| Job Classification Accuracy | 0.990 |
| Career Switcher Accuracy | 0.959 |
| Identity Swap Top-10 Change | 0.0% |

> These metrics are based on synthetic data and are not validated real-world recruitment performance.

## 🛠️ Tech Stack

- Python
- Flask
- Scikit-learn
- NLP / Text Processing
- TF-IDF
- Multi-Layer Perceptron (MLP)
- Machine Learning
- Resume Parsing (PDF, DOCX, TXT)

## 📁 Project Structure

```text
EviProfile/
├── app.py
├── run_pipeline.py
├── requirements.txt
├── resume_intel/
├── tests/
├── data/
├── results/
└── models/
```

*The folder structure above is illustrative. Actual filenames may differ.*

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/EviProfile.git
cd EviProfile
```

### Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

Start the Flask application:

```bash
python app.py
```

Run the ML pipeline (if the entry-point file exists):

```bash
python run_pipeline.py
```

*Update these commands if your actual entry-point filenames are different.*

## 🔮 Future Improvements

- Evaluate on real-world anonymized resumes.
- Improve resume parsing accuracy.
- Add OCR support for scanned resumes.
- Support multilingual resumes.
- Improve model fairness and explainability.
- Expand candidate ranking evaluation.
- Add more real-world validation and testing.

## ⚠️ Limitations

- Evaluation currently uses synthetic resume data.
- OCR for scanned resumes is not supported.
- Non-English resume processing is not currently supported.
- Real-world hiring performance has not been established.
- Scoring and ranking should be validated for fairness and reliability before deployment.

## 🤝 Responsible Use

EviProfile is an experimental resume analysis and candidate profiling system. Its scores and rankings should be treated as decision-support information, not as the sole basis for hiring decisions. Human review, fairness evaluation, privacy protection, and real-world validation are essential before practical deployment.

---

**Project Name:** EviProfile  
**Project Type:** AI / Machine Learning + NLP  
**Purpose:** Automated Candidate Profiling and Resume Intelligence Extraction
