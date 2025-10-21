# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install build tools, gdown, and Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt gdown && \
    python -m nltk.downloader stopwords

# Copy the rest of the application code into the container
COPY . .

# Create the models directory and download the models using gdown
RUN mkdir -p models && \
    gdown --id 12dkExGVsoqpNjGq-meBO5uF7nXrPOvtu -O models/w2v_classifier.pkl && \
    gdown --id 1a2stf2DF8G9_YmbSDgtm3qlVOlGm1Gpk -O models/word2vec_model.model

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to run the app
# We use 0.0.0.0 to make the app accessible from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
