# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies, including curl
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install build tools and Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    python -m nltk.downloader stopwords

# Copy the rest of the application code into the container
COPY . .

# Create the models directory and download the models
RUN mkdir -p models && \
    curl -L 'https://drive.google.com/uc?export=download&id=12dkExGVsoqpNjGq-meBO5uF7nXrPOvtu' -o 'models/w2v_classifier.pkl' && \
    curl -L 'https://drive.google.com/uc?export=download&id=1a2stf2DF8G9_YmbSDgtm3qlVOlGm1Gpk' -o 'models/word2vec_model.model'

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to run the app
# We use 0.0.0.0 to make the app accessible from outside the container
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
