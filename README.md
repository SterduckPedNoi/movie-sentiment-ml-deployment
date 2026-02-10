# 🎬 Movie Sentiment Analysis API

A **Machine Learning-powered Movie Review Sentiment Analysis API** built with FastAPI and deployed on Render.

Submit a movie review text and receive a **Positive or Negative sentiment prediction** in real time.

---

## 🚀 Project Overview

This project demonstrates how to build, serve, and deploy a Machine Learning model as a web API using modern backend tools.

It covers:
- Machine Learning model inference
- REST API development with FastAPI
- Cloud deployment on Render (without Docker)

---

## ✨ Features

- 🧠 ML-based sentiment classification  
- ⚡ FastAPI backend with automatic API documentation  
- 🌍 Cloud deployment on Render  
- 📡 RESTful API ready for integration  

---

## 🛠 Tech Stack

| Layer | Technology |
|------|------------|
| Language | Python |
| ML | Scikit-learn / NLP Model |
| Backend | FastAPI |
| Server | Uvicorn |
| Cloud | Render |

---

## 📂 Project Structure

movie-sentiment-ml-deployment/
│
├── app/ # FastAPI application
├── model/ # Trained ML model files
├── requirements.txt # Python dependencies
├── runtime.txt # Python runtime version 
└── README.md

---

## ⚙️ Local Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/SterduckPedNoi/movie-sentiment-ml-deployment.git
cd movie-sentiment-ml-deployment
2️⃣ Install Dependencies
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows

pip install -r requirements.txt

▶️ Run Locally
uvicorn app.main:app --reload


Open Swagger UI at:
👉 http://localhost:8000/docs

☁️ Deployment on Render (No Docker)

This project is deployed on Render using a native Python environment.

🚀 Deployment Steps

Go to https://render.com
 and create an account

Click New → Web Service

Connect your GitHub repository

Choose Python as the environment

⚙️ Render Configuration

Set the following values:

Build Command

pip install -r requirements.txt


Start Command

uvicorn app.main:app --host 0.0.0.0 --port 10000


Python Version

Defined in runtime.txt

🌍 Access the API

After deployment, Render will provide a public URL:

API Base URL

https://your-service-name.onrender.com


Swagger Documentation

https://your-service-name.onrender.com/docs

🧊 Cold Start Notice

Free-tier Render services may sleep during inactivity.
The first request after a period of inactivity may take a few seconds.

📡 API Reference
POST /predict
Request
{
  "text": "This movie was absolutely amazing!"
}

Response
{
  "prediction": "Positive",
  "confidence": 0.92
}

🎯 Use Cases

Movie review sentiment analysis

NLP API demo project

Machine Learning deployment portfolio

Backend service for AI applications

👨‍💻 Author

SterduckPedNoi
GitHub: https://github.com/SterduckPedNoi
