import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import gensim
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import distance
from fuzzywuzzy import fuzz
from tqdm import tqdm

# Download required NLTK data
nltk.download('stopwords', quiet=True)

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

def train_and_save_models():
    print("Loading and preprocessing data...")
    
    # Load data
    df = pd.read_csv("./quora-question-pairs/train.csv")
    df = df[["question1", "question2", "is_duplicate"]].dropna()
    
    # Balance dataset
    df_0 = df[df["is_duplicate"] == 0].sample(149263, random_state=42)
    df_1 = df[df["is_duplicate"] == 1].sample(149263, random_state=42)
    df = pd.concat([df_0, df_1]).reset_index(drop=True)
    
    # Preprocess questions
    df["question1"] = df["question1"].apply(preprocess)
    df["question2"] = df["question2"].apply(preprocess)
    
    print("Training Word2Vec model...")
    
    # Train Word2Vec
    questions = list(df["question1"]) + list(df["question2"])
    ques_sent = [gensim.utils.simple_preprocess(sentence) for sentence in questions]
    
    model = gensim.models.Word2Vec(window=5, min_count=3, sg=0, vector_size=100)
    model.build_vocab(ques_sent)
    model.train(corpus_iterable=ques_sent, total_examples=model.corpus_count, epochs=model.epochs)
    
    print("Extracting features...")
    
    def document_vector(doc):
        doc = [word for word in doc.split() if word in model.wv.index_to_key]
        if len(doc) == 0:
            return np.zeros(model.vector_size)
        return np.mean(model.wv[doc], axis=0)
    
    # Extract W2V features
    X_list = [document_vector(doc) for doc in tqdm(df["question1"].values)]
    y_list = [document_vector(doc) for doc in tqdm(df["question2"].values)]
    
    w2v_features = np.concatenate([np.array(X_list), np.array(y_list)], axis=1)
    
    # Extract manual features
    df1 = df[["question1", "question2"]].copy()
    df1["q1_len"] = df1["question1"].str.len()
    df1["q2_len"] = df1["question2"].str.len()
    df1["q1_num_words"] = df1["question1"].apply(lambda x: len(str(x).split()))
    df1["q2_num_words"] = df1["question2"].apply(lambda x: len(str(x).split()))
    
    # Common words features
    def common_words(row):
        w1 = set(str(row["question1"]).split())
        w2 = set(str(row["question2"]).split())
        return len(w1 & w2)
    
    def word_total(row):
        return len(str(row['question1']).split()) + len(str(row['question2']).split())
    
    df1["word_common"] = df1.apply(common_words, axis=1)
    df1["word_total"] = df1.apply(word_total, axis=1)
    df1["word_share"] = df1.apply(lambda x: round(x["word_common"] / x["word_total"], 2) if x["word_total"] > 0 else 0, axis=1)
    
    # Token features
    def fetch_token_features(row):
        q1, q2 = row['question1'], row['question2']
        SAFE_DIV, STOP_WORDS = 0.0001, stopwords.words("english")
        
        q1_tokens, q2_tokens = q1.split(), q2.split()
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return [0.0] * 8
        
        q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
        q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
        q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
        q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
        
        common_word_count = len(q1_words.intersection(q2_words))
        common_stop_count = len(q1_stops.intersection(q2_stops))
        common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
        
        return [
            common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV),
            common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV),
            common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV),
            common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV),
            common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV),
            common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV),
            int(q1_tokens[-1] == q2_tokens[-1]),
            int(q1_tokens[0] == q2_tokens[0])
        ]
    
    token_features = df1.apply(fetch_token_features, axis=1)
    for i, col in enumerate(["cwc_min", "cwc_max", "csc_min", "csc_max", "ctc_min", "ctc_max", "last_word_eq", "first_word_eq"]):
        df1[col] = [x[i] for x in token_features]
    
    # Length features
    def fetch_length_features(row):
        q1, q2 = row['question1'], row['question2']
        q1_tokens, q2_tokens = q1.split(), q2.split()
        
        if len(q1_tokens) == 0 or len(q2_tokens) == 0:
            return [0.0] * 3
        
        abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
        mean_len = (len(q1_tokens) + len(q2_tokens)) / 2
        strs = list(distance.lcsubstrings(q1, q2))
        longest_substr_ratio = len(strs[0]) / (min(len(q1), len(q2)) + 1) if strs else 0
        
        return [abs_len_diff, mean_len, longest_substr_ratio]
    
    length_features = df1.apply(fetch_length_features, axis=1)
    for i, col in enumerate(['abs_len_diff', 'mean_len', 'longest_substr_ratio']):
        df1[col] = [x[i] for x in length_features]
    
    # Fuzzy features
    def fetch_fuzzy_features(row):
        q1, q2 = row['question1'], row['question2']
        return [
            fuzz.QRatio(q1, q2),
            fuzz.partial_ratio(q1, q2),
            fuzz.token_sort_ratio(q1, q2),
            fuzz.token_set_ratio(q1, q2)
        ]
    
    fuzzy_features = df1.apply(fetch_fuzzy_features, axis=1)
    for i, col in enumerate(['fuzz_ratio', 'fuzz_partial_ratio', 'token_sort_ratio', 'token_set_ratio']):
        df1[col] = [x[i] for x in fuzzy_features]
    
    # Combine features
    manual_features = df1[["q1_len", "q2_len", "q1_num_words", "q2_num_words", "word_common", "word_total", "word_share", 
                          "cwc_min", "cwc_max", "csc_min", "csc_max", "ctc_min", "ctc_max", "last_word_eq", "first_word_eq", 
                          "abs_len_diff", "mean_len", "longest_substr_ratio", "fuzz_ratio", "fuzz_partial_ratio", 
                          "token_sort_ratio", "token_set_ratio"]].values
    
    X = np.concatenate([w2v_features, manual_features], axis=1)
    y = df["is_duplicate"].values
    
    print("Training Random Forest classifier...")
    
    # Train classifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    print("Saving models...")
    
    # Save models
    with open('models/w2v_classifier.pkl', 'wb') as f:
        pickle.dump(rf, f)
    
    model.save('models/word2vec_model.model')
    
    print("Models saved successfully!")
    print(f"Test accuracy: {rf.score(X_test, y_test):.3f}")

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    train_and_save_models()