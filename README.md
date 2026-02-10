# 🎬 Movie Sentiment Analysis API

A Machine Learning-powered API for analyzing movie review sentiment, built with FastAPI and deployed on Render.

Send a movie review as text and receive a real-time sentiment prediction: **Positive** or **Negative**.

---

## 🚀 Overview

This project demonstrates a complete Machine Learning deployment workflow, from model inference to cloud deployment using FastAPI.

It is designed for learning purposes, portfolio showcasing, and real-world API usage.

---

## ✨ Features

- 🧠 Machine Learning-based sentiment classification  
- ⚡ FastAPI backend with automatic Swagger documentation  
- 🌍 Deployed on Render (Python environment, no Docker)  
- 📡 RESTful API ready for integration  

---

## 🛠 Tech Stack

| Layer | Technology |
|------|------------|
| Language | Python / JavaScript / CSS / HTML |
| Machine Learning | Scikit-learn / NLP Model |
| Backend | FastAPI |
| Server | Uvicorn |
| Cloud Platform | Render |

---

## 📂 Project Structure

```
movie-sentiment-ml-deployment/
│
├── backend/                    # FastAPI backend, ML inference & frontend files
│   ├── main.py                 # FastAPI entry point
│   ├── download_models.py      # Download ML models from this GitHub repository
│   ├── requirements.txt        # Backend dependencies
│   ├── runtime.txt             # Python version for Render
│   │
│   ├── models/                 # Trained ML models
│   │   ├── v1_baseline/
│   │   ├── v2_error_boost/
│   │   ├── v3_improved_tfidf/
│   │   ├── v4_linear_svm/
│   │   └── v5_ensemble/
│   │
│   └── frontend/               # Frontend (HTML / CSS / JavaScript)
│       ├── index.html
│       ├── script.js
│       └── style.css
│
└── README.md
```

> The frontend is served from the `backend/frontend/` directory  
> while all ML models are managed under `backend/models/`.

---

## ⚙️ Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/SterduckPedNoi/movie-sentiment-ml-deployment.git
cd movie-sentiment-ml-deployment
```

---

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
uvicorn app.main:app --reload
```

Open Swagger UI at:  
http://localhost:8000/docs

---

## ☁️ Deployment on Render (No Docker)

This application is deployed on Render using a native Python environment.

### Deployment Steps

1. Create an account at https://render.com  
2. Click **New → Web Service**  
3. Connect your GitHub repository  
4. Select **Python** as the environment  

---

### Render Configuration

Set the following commands in Render:

**Root Directory**
```bash
backend
```

**Build Command**
```bash
pip install -r requirements.txt && python download_models.py
```

**Start Command**
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Python Version**
```
Defined in runtime.txt
```

---

### Access the API

After deployment, Render will provide a public URL:

- API Base URL  
  ```
  https://your-service-name.onrender.com
  ```

- Swagger UI  
  ```
  https://your-service-name.onrender.com/docs
  ```

---

### Cold Start Notice

Render free-tier services may enter sleep mode when inactive.  
The first request after inactivity may take a few seconds to respond.

---

## 📡 API Reference

### POST `/predict`

**Request Body**
```json
{
  "text": "This movie was absolutely amazing!"
}
```

**Response**
```json
{
  "prediction": "Positive",
  "confidence": 0.92
}
```

---

## 🎯 Use Cases

- Movie review sentiment analysis  
- NLP API demo project  
- Machine Learning deployment portfolio  
- Backend service for AI applications  

---

## 👨‍💻 Author

SterduckPedNoi  
GitHub: https://github.com/SterduckPedNoi

---

## ❤️ Built with FastAPI and Machine Learning
