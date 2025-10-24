---
title: Duplicate Question Detector
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.1
app_file: app.py
pinned: false
license: others
---

# 🔍 Duplicate Question Detector

A machine learning application that detects whether two questions are duplicates using Word2Vec embeddings and advanced feature engineering techniques.

## 📋 Overview

This project implements a sophisticated duplicate question detection system that combines:
- **Word2Vec embeddings** for semantic understanding
- **Advanced feature engineering** including fuzzy matching, token analysis, and length features
- **Random Forest classifier** for robust prediction
- **Interactive Streamlit interface** for easy testing

## 🎯 Features

- **Semantic Analysis**: Uses Word2Vec to capture semantic similarity between questions
- **Feature Engineering**: 22+ handcrafted features including:
  - Token-based features (common words, stop words)
  - Length-based features (character/word counts, ratios)
  - Fuzzy matching features (edit distance, partial ratios)
  - Linguistic features (first/last word matching)
- **Real-time Prediction**: Interactive web interface with confidence scores
- **Preprocessing Pipeline**: Comprehensive text cleaning and normalization

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/duplicate-question-detector.git
cd duplicate-question-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the [Quora Question Pairs dataset](https://www.kaggle.com/c/quora-question-pairs) and place it in `./quora-question-pairs/train.csv`

4. Train the models:
```bash
python training_models.py
```

5. Run the Streamlit app:
```bash
streamlit run app.py
```

## 📊 Model Architecture

### Feature Engineering Pipeline

The model uses a comprehensive feature extraction pipeline:

1. **Text Preprocessing**:
   - Lowercasing and whitespace normalization
   - Special character replacement (%, $, €, @)
   - Number normalization (1000 → 1k, 1000000 → 1m)
   - Contraction expansion (don't → do not)
   - HTML tag removal
   - Punctuation removal
   - Porter stemming

2. **Word2Vec Features** (200 dimensions):
   - Question 1 embedding (100D)
   - Question 2 embedding (100D)

3. **Manual Features** (22 dimensions):
   - **Basic features**: Length, word count
   - **Common word features**: Shared words, ratios
   - **Token features**: Word/stop word overlap ratios
   - **Length features**: Absolute difference, mean length
   - **Fuzzy features**: Edit distance, partial ratios
   - **Linguistic features**: First/last word matching

### Model Training

- **Algorithm**: Random Forest Classifier (100 trees)
- **Dataset**: [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) - Balanced dataset (298,526 samples)
- **Train/Test Split**: 80/20 stratified split
- **Evaluation**: Accuracy, precision, recall, F1-score

## 🎮 Usage

### Web Interface

1. Launch the app: `streamlit run app.py`
2. Enter two questions in the text areas
3. Click "Check for Duplicates"
4. View results with confidence scores

### Programmatic Usage

```python
from app import predict_duplicate, load_models, download_nltk_data

# Load models
stop_words = download_nltk_data()
classifier, w2v_model = load_models()

# Predict
q1 = "What is machine learning?"
q2 = "Can you explain machine learning?"
prediction, probabilities = predict_duplicate(q1, q2, classifier, w2v_model, stop_words)

print(f"Duplicate: {prediction}")
print(f"Confidence: {probabilities[1]*100:.1f}%")
```

## 📈 Performance

The model achieves strong performance on the Quora Question Pairs dataset:
- **Accuracy**: ~85%+ on test set
- **Features**: 222 total features (200 Word2Vec + 22 engineered)
- **Processing**: Real-time inference (<1 second per pair)

## 🔧 Technical Details

### Dependencies

- **Core ML**: scikit-learn, gensim, numpy, pandas
- **NLP**: nltk, beautifulsoup4
- **String Matching**: fuzzywuzzy, python-levenshtein, distance
- **Web App**: streamlit
- **Utilities**: tqdm, pickle

### File Structure

```
├── app.py                 # Streamlit web application
├── training_models.py     # Model training script
├── requirements.txt       # Python dependencies
├── models/
│   ├── w2v_classifier.pkl # Trained Random Forest model
│   └── word2vec_model.model # Trained Word2Vec model
└── README.md             # This file
```

### Local Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (if not already trained)
python training_models.py

# Run the app
streamlit run app.py
```

## 📞 Contact

- **Author**: Priyansh Yadav
- **GitHub**: [@ipriyanshyadav](https://github.com/ipriyanshyadav)
---

*Built with ❤️ using Python, Word2Vec, and Streamlit*
