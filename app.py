import streamlit as st
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import gensim
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import distance
from fuzzywuzzy import fuzz
import os
import requests
from pathlib import Path

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('stopwords', quiet=True)
    return stopwords.words("english")

def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            break
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

def ensure_models_exist():
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    classifier_path = models_dir / 'w2v_classifier.pkl'
    w2v_model_path = models_dir / 'word2vec_model.bin'
    
    if not classifier_path.exists():
        st.info("Downloading classifier model...")
        download_file_from_google_drive('12dkExGVsoqpNjGq-meBO5uF7nXrPOvtu', classifier_path)
    
    if not w2v_model_path.exists():
        st.info("Downloading Word2Vec model...")
        download_file_from_google_drive('1a2stf2DF8G9_YmbSDgtm3qlVOlGm1Gpk', w2v_model_path)

# Load models
@st.cache_resource
def load_models():
    ensure_models_exist()
    
    with open('models/w2v_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    
    w2v_model = gensim.models.Word2Vec.load('models/word2vec_model.bin')
    
    return classifier, w2v_model

def preprocess(q):
    q = str(q).lower().strip()
    
    # Replace special characters
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')
    
    # Replace numbers
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)
    
    # Contractions
    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "can not", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
        "hasn't": "has not", "haven't": "have not", "he's": "he is", "i'm": "i am",
        "i've": "i have", "isn't": "is not", "it's": "it is", "let's": "let us",
        "she's": "she is", "that's": "that is", "there's": "there is", "they're": "they are",
        "they've": "they have", "we're": "we are", "we've": "we have", "weren't": "were not",
        "what's": "what is", "where's": "where is", "who's": "who is", "won't": "will not",
        "wouldn't": "would not", "you're": "you are", "you've": "you have"
    }
    
    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
    
    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Remove HTML tags
    q = BeautifulSoup(q, 'html.parser').get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()
    
    # Stemming
    stemmer = PorterStemmer()
    q = ' '.join([stemmer.stem(word) for word in q.split()])
    
    return q

def get_w2v_features(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

def predict_duplicate(q1, q2, classifier, w2v_model, stop_words):
    # Preprocess questions
    q1_processed = preprocess(q1)
    q2_processed = preprocess(q2)
    
    # W2V features
    q1_w2v = get_w2v_features(q1_processed, w2v_model)
    q2_w2v = get_w2v_features(q2_processed, w2v_model)
    
    # Manual features
    q1_len = len(q1_processed)
    q2_len = len(q2_processed)
    q1_num_words = len(q1_processed.split())
    q2_num_words = len(q2_processed.split())
    
    # Common words
    w1 = set(q1_processed.split())
    w2 = set(q2_processed.split())
    word_common = len(w1 & w2)
    word_total = len(q1_processed.split()) + len(q2_processed.split())
    word_share = round(word_common / word_total, 2) if word_total > 0 else 0
    
    # Token features
    SAFE_DIV = 0.0001
    q1_tokens = q1_processed.split()
    q2_tokens = q2_processed.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        token_features = [0.0] * 8
    else:
        q1_words = set([word for word in q1_tokens if word not in stop_words])
        q2_words = set([word for word in q2_tokens if word not in stop_words])
        q1_stops = set([word for word in q1_tokens if word in stop_words])
        q2_stops = set([word for word in q2_tokens if word in stop_words])
        
        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stops.intersection(q2_stops))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
        
        token_features = [
            common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV),
            common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV),
            common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV),
            common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV),
            common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV),
            common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV),
            int(q1_tokens[-1] == q2_tokens[-1]) if q1_tokens and q2_tokens else 0,
            int(q1_tokens[0] == q2_tokens[0]) if q1_tokens and q2_tokens else 0
        ]
    
    # Length features
    abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
    mean_len = (len(q1_tokens) + len(q2_tokens)) / 2
    strs = list(distance.lcsubstrings(q1_processed, q2_processed))
    longest_substr_ratio = len(strs[0]) / (min(len(q1_processed), len(q2_processed)) + 1) if strs else 0
    
    # Fuzzy features
    fuzz_ratio = fuzz.QRatio(q1_processed, q2_processed)
    fuzz_partial_ratio = fuzz.partial_ratio(q1_processed, q2_processed)
    token_sort_ratio = fuzz.token_sort_ratio(q1_processed, q2_processed)
    token_set_ratio = fuzz.token_set_ratio(q1_processed, q2_processed)
    
    # Combine features
    w2v_features = np.concatenate([q1_w2v, q2_w2v])
    manual_features = np.array([
        q1_len, q2_len, q1_num_words, q2_num_words, word_common, word_total, word_share,
        *token_features, abs_len_diff, mean_len, longest_substr_ratio,
        fuzz_ratio, fuzz_partial_ratio, token_sort_ratio, token_set_ratio
    ])
    
    features = np.concatenate([w2v_features, manual_features]).reshape(1, -1)
    
    # Make prediction
    pred = classifier.predict(features)[0]
    prob = classifier.predict_proba(features)[0]
    
    return pred, prob

def main():
    st.set_page_config(
        page_title="Duplicate Question Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    st.markdown("<h1 style='text-align: center;'>🔍 Duplicate Question Detector</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Detect if two questions are duplicates using Word2Vec and Advanced Feature Engineering</h3>", unsafe_allow_html=True)
    st.markdown("")
    st.divider()
    
    # Load models and data
    stop_words = download_nltk_data()
    classifier, w2v_model = load_models()
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='text-align: center;'>Question 1</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Enter the first question:</p>", unsafe_allow_html=True)
        q1 = st.text_area("", value=st.session_state.get('example_q1', ''), height=100, key="q1", label_visibility="collapsed")
    
    with col2:
        st.markdown("<h3 style='text-align: center;'>Question 2</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Enter the second question:</p>", unsafe_allow_html=True)
        q2 = st.text_area("", value=st.session_state.get('example_q2', ''), height=100, key="q2", label_visibility="collapsed")
    
    # Predict button
    if st.button("🔍 Check for Duplicates", type="primary", use_container_width=True):
        if q1.strip() and q2.strip():
            with st.spinner("Analyzing questions..."):
                pred, prob = predict_duplicate(q1, q2, classifier, w2v_model, stop_words)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if pred == 1:
                        st.success("✅ **DUPLICATE**")
                        confidence = prob[1] * 100
                    else:
                        st.error("❌ **NOT DUPLICATE**")
                        confidence = (1 - prob[1]) * 100
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    st.metric("Duplicate Probability", f"{prob[1]*100:.1f}%")
                
                # Progress bar for confidence
                st.progress(confidence/100)
        
        else:
            st.warning("⚠️ Please enter both questions!")
    
    # Example questions
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>💡 Try These Examples</h3>", unsafe_allow_html=True)
    
    examples = [
        ("What is the capital of India?", "What is the capital of India?", "Identical questions"),
        ("How do I learn Python?", "How can I learn Python programming?", "Similar meaning"),
        ("What is machine learning?", "How does a car engine work?", "Different topics"),
        ("How to lose weight fast?", "What are quick ways to reduce weight?", "Similar meaning"),
        ("What is the best programming language?", "Which programming language should I learn first?", "Related but different")
    ]
    
    cols = st.columns(5)
    for i, (ex_q1, ex_q2, desc) in enumerate(examples):
        with cols[i]:
            if st.button(f"📝 **Example {i+1}**\n\n{desc}", key=f"ex_{i}", use_container_width=True):
                st.session_state.example_q1 = ex_q1
                st.session_state.example_q2 = ex_q2
                st.rerun()

if __name__ == "__main__":
    main()