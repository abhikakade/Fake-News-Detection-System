# 📰 Fake News Detection System: An End-to-End ML Project

This repository showcases my journey in Machine Learning, specifically focusing on **Natural Language Processing (NLP)**. The project covers the entire lifecycle of a data product: from exploratory data analysis and model training to deploying a cloud-based API and building a functional user interface.

## 🚀 Project Overview
The goal of this project is to classify news articles as **"Real"** or **"Fake"** based on their linguistic patterns. It uses a Logistic Regression model trained on TF-IDF features to identify subtle markers of misinformation.

### Key Features:
* **Custom Text Preprocessing:** Implements NLTK-based tokenization, stopword removal, and lemmatization.
* **Feature Engineering:** Utilizes Term Frequency-Inverse Document Frequency (TF-IDF) to convert text into numerical vectors.
* **Cloud Deployment:** Model served via AWS Lambda for scalable, serverless inference.
* **Interactive UI:** A React-based web application allowing users to verify news snippets in real-time.

---

## 🛠️ Technical Stack
* **ML/NLP:** Python, Scikit-learn, Pandas, NLTK, Joblib.
* **Backend:** AWS Lambda, Serverless Python.
* **Frontend:** React.js, CSS3.
* **DevOps:** Model Serialization (Pickle/Joblib), CORS handling.

---

## 📂 File Structure
* **`FakeNewsDetection_TFID.ipynb`**: The development core. Includes data cleaning, visualization of top "fake news" keywords, model training, and performance evaluation.
* **`handler.py`**: The production script for AWS Lambda. It loads the serialized model (`lr_model.pkl`) and vectorizer, processes incoming JSON requests, and returns classification probabilities.
* **`App.js`**: The React frontend component that connects to the API gateway and displays the prediction results with a confidence score.

---

## 🧠 What I Learned
1.  **NLP Pipeline Construction:** How to transform messy, raw text into structured data that a machine can understand.
2.  **Model Persistence:** Learning how to save models and vectorizers using `joblib` so they can be reused without retraining.
3.  **Inference Engineering:** Designing a backend that can handle text preprocessing in the same way the model was trained (essential for accuracy).
4.  **Full-Stack ML:** Connecting a frontend to a machine learning model via a REST API.

---

## ⚙️ How to Run
1.  **Training:** Run the Jupyter Notebook to generate `lr_model.pkl` and `tfidf_vectorizer.pkl`.
2.  **Backend:** Deploy `handler.py` (and artifacts) to an AWS Lambda function and set up an API Gateway.
3.  **Frontend:** * Replace the `API_ENDPOINT` in `App.js` with your Lambda URL.
    * Run `npm start` to launch the validator.

---

*This project was built to demonstrate proficiency in building practical, end-to-end machine learning applications.*
