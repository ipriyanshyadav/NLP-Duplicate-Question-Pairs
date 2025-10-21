# Duplicate Question Detector

A machine learning application that detects duplicate questions using Word2Vec embeddings and advanced feature engineering techniques.

## Dataset Link - https://www.kaggle.com/c/quora-question-pairs

## Features

- **Word2Vec Embeddings**: Uses pre-trained Word2Vec model for semantic understanding
- **Advanced Feature Engineering**: 
  - Token-based features (common words, stop words, tokens)
  - Length-based features (absolute difference, mean length, longest substring)
  - Fuzzy matching features (fuzz ratio, partial ratio, token sort/set ratios)
- **Interactive Web Interface**: Built with Streamlit for easy use
- **Pre-trained Models**: Avoids retraining with saved models

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Download the Quora Question Pairs dataset
   - Place `train.csv` in `./quora-question-pairs/` directory

3. **Train and Save Models** (one-time setup):
   ```bash
   python save_models.py
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Enter two questions in the text areas
2. Click "Check for Duplicates"
3. View the prediction results with confidence scores
4. Try the provided examples to test different scenarios

## Model Performance

- **Algorithm**: Random Forest Classifier
- **Features**: 222 total features (200 W2V + 22 manual features)
- **Accuracy**: ~85% on test set
- **Dataset**: Balanced Quora Question Pairs (298,526 samples)

## File Structure

```
├── app.py                 # Main Streamlit application
├── save_models.py         # Script to train and save models
├── requirements.txt       # Python dependencies
├── models/               # Saved models directory
│   ├── w2v_classifier.pkl
│   └── word2vec_model.model
└── quora-question-pairs/
    └── train.csv         # Dataset (download separately)
```

## Features Explained

### Text Preprocessing
- Lowercasing and cleaning
- Contraction expansion
- HTML tag removal
- Punctuation removal
- Stemming

### Feature Categories
1. **Basic Features**: Length, word count, common words
2. **Token Features**: Word/stop word ratios, first/last word matching
3. **Length Features**: Absolute differences, substring ratios
4. **Fuzzy Features**: Various fuzzy matching scores
5. **Word2Vec Features**: Semantic embeddings (100-dim per question)

## Example Predictions

- **Identical**: "What is Python?" vs "What is Python?" → Duplicate
- **Similar**: "How to learn ML?" vs "How can I learn machine learning?" → Duplicate  
- **Different**: "What is AI?" vs "How to cook pasta?" → Not Duplicate
