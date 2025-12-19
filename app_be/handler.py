import json
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import logging

# --- 1. Global Initialization (Model and Preprocessor Loading) ---
# This code runs only once during the 'cold start' of the container.

# Set up basic logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Define file paths for the artifacts inside the container
MODEL_PATH = "lr_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Point NLTK to the directory where data was downloaded during build
nltk.data.path.append(os.environ.get("LAMBDA_TASK_ROOT", "") + "/nltk_data")

# Initialize NLTK components for preprocessing
try:
    # Attempt to load NLTK resources (must be packaged in the Docker image)
    STOPWORDS = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
except LookupError:
    # If NLTK data is not available (common in slim containers), handle here
    logger.error(
        "NLTK resources not found. Ensure nltk data is included in the Docker build."
    )
    # Fallback to an empty set if NLTK data is missing, or rely on TfidfVectorizer's built-in stop words.
    STOPWORDS = set()

# Load the artifacts
try:
    # We use joblib as it handles scikit-learn objects efficiently
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logger.info("Model and Vectorizer loaded successfully.")
except Exception as e:
    model = None
    vectorizer = None
    logger.error(f"Error loading model artifacts: {e}")

# --- 2. Preprocessing Function (Replicates Training Prep) ---


def clean_text(text):
    """Applies the exact cleaning and normalization steps used during training."""
    if not text:
        return ""

    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize, remove stopwords, and lemmatize
    text = " ".join(
        [lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS]
    )
    return text


# --- 3. The Lambda Handler Function ---


def lambda_handler(event, context):
    """
    AWS Lambda function handler for API Gateway requests.
    Expects a JSON body with a 'text' field containing the news article content.
    """

    if model is None or vectorizer is None:
        return {
            "statusCode": 503,
            "body": json.dumps({"error": "Model initialization failed."}),
        }

    try:
        # API Gateway sends the body as a string, so we must parse it
        # print("event", event)

        if isinstance(event, str):
            data = json.loads(event)
        else:
            data = event

        raw_text = data.get("text", "")
        if not raw_text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'text' field in request body."}),
            }

        # 1. Clean the input text
        cleaned_text = clean_text(raw_text)

        # 2. Transform the text using the fitted TF-IDF vectorizer
        # We wrap the text in a list for the vectorizer
        text_vectorized = vectorizer.transform([cleaned_text])

        # 3. Predict the label (0: Real, 1: Fake)
        prediction_label = model.predict(text_vectorized)[0].item()

        # 4. Get the probabilities
        probabilities = model.predict_proba(text_vectorized)[0].tolist()

        # Create a common headers dictionary to reuse
        CORS_HEADERS = {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # For production, replace * with your React app's URL
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        }

        # Format the response
        response = {
            "is_fake": bool(prediction_label),
            "prediction_label": int(prediction_label),
            "probabilities": {
                "real": round(probabilities[0], 4),
                "fake": round(probabilities[1], 4),
            },
            "interpretation": "A higher probability for 'fake' suggests the presence of linguistic features strongly correlated with fake news in the training data (e.g., highly positive model coefficients).",
        }

        return {
            "statusCode": 200,
            "headers": CORS_HEADERS,
            "body": json.dumps(response),
        }

    except KeyError:
        return {
            "statusCode": 400,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": "Invalid JSON format or missing keys."}),
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": f"Internal server error: {str(e)}"}),
        }
