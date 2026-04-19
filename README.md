# Clinical Appointment No-Show Prediction & Agentic Care Coordination System

## 📌 Overview
This project predicts the probability that a patient will miss a scheduled clinical appointment (No-Show) using traditional machine learning techniques, and provides an **AI Care Coordinator** to generate actionable, evidence-based intervention plans using LangGraph and Groq APIs.

The single Streamlit application (`app.py`) provides both:
1. **ML Predictions:** Identifies and visualizes probability of no-shows.
2. **AI Care Coordinator:** Leverages RAG (Retrieval-Augmented Generation) with FAISS to analyze high-risk cohorts and recommend interventions against local health guidelines.

---

## 🎯 Problem Statement
Missed medical appointments lead to revenue loss, inefficient scheduling, and wasted medical resources. This system helps clinics proactively identify high-risk appointments and takes preventive action through an agentic AI system that generates structured reports.

---

## 🛠 Milestones Included
- **Milestone 1:** Traditional ML pipeline with Random Forest Classifier and risk categorization.
- **Milestone 2:** LangGraph agent featuring RAG pipeline, retrieving from CDC/WHO guidelines, generating reports via `llama-3.1-8b-instant` (via Groq), and exporting to valid PDF.

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Imposon/Clinical_No_show.git
cd Clinical_No_show
```

### 2️⃣ Create a Virtual Environment

Make sure Python 3.10 or above is installed.
```bash
python3 -m venv venv
```
Activate the environment:
Mac/Linux
```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Setup Environment Variables & Build Index
You will need a free [Groq API Key](https://console.groq.com/).

Run the FAISS index builder locally once:
```bash
python build_faiss_index.py
```
This will compile guidelines into a local `faiss_index/` folder, which should be committed to GitHub.

Create a `.streamlit/secrets.toml` file in your directory to store your Groq API Key:
```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxx"
```

### 5️⃣ Run the Application
```bash
streamlit run app.py
```
The app will open in your browser at `http://localhost:8501`.

---

## ☁️ Deployment Notes (Streamlit Community Cloud)
1. Get an API key from the Groq console.
2. Make sure you ran `python build_faiss_index.py` locally and successfully pushed the resulting `faiss_index/` folder to your GitHub repository.
3. Deploy the application straight from your GitHub repository onto Streamlit Community Cloud.
4. Go to App Settings -> **Secrets** and add:
   ```toml
   GROQ_API_KEY = "gsk_xxxxxxx"
   ```
5. Deploy and restart!
