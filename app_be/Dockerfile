# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# 1. Copy requirements first to leverage Docker caching
COPY requirements.txt .

# 2. Install Python dependencies (This makes 'nltk' available for the next step) [cite: 1]
# We install directly to ${LAMBDA_TASK_ROOT} which is /var/task
RUN pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# 3. Download NLTK data to a specific folder inside the task root
RUN python -m nltk.downloader -d "${LAMBDA_TASK_ROOT}/nltk_data" wordnet stopwords omw-1.4

# 4. Copy the serialized model artifacts and inference handler [cite: 3]
COPY lr_model.pkl ${LAMBDA_TASK_ROOT}
COPY tfidf_vectorizer.pkl ${LAMBDA_TASK_ROOT}
COPY handler.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler function
CMD ["handler.lambda_handler"]