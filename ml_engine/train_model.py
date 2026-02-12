import pandas as pd
import numpy as np
import pickle
import os
import sys
import re

# ----------------- Configuration -----------------
# DATA_PATH: Path to the dataset folder
DATA_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DATA_PATH, 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(DATA_PATH, 'vectorizer.pkl')

def print_section(title):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

def load_data():
    print_section("STEP 1: Data Loading")
    
    # Check for Kaggle Dataset (True.csv and Fake.csv)
    true_path = os.path.join(DATA_PATH, '../data/True.csv')
    fake_path = os.path.join(DATA_PATH, '../data/Fake.csv')
    
    if os.path.exists(true_path) and os.path.exists(fake_path):
        print(f"[INFO] Detected Kaggle Dataset (ISOT) at {DATA_PATH}/../data/")
        print("   - Loading True.csv...")
        df_true = pd.read_csv(true_path)
        df_true['label'] = 1 # Real
        
        print("   - Loading Fake.csv...")
        df_fake = pd.read_csv(fake_path)
        df_fake['label'] = 0 # Fake
        
        df = pd.concat([df_true, df_fake], axis=0)
        print(f"[SUCCESS] Loaded {len(df)} articles.")
    else:
        # Fallback to local small dataset
        dataset_path = os.path.join(DATA_PATH, '../dataset.csv')
        if os.path.exists(dataset_path):
             print(f"[INFO] Using local dataset: {dataset_path}")
             df = pd.read_csv(dataset_path)
             print(f"[SUCCESS] Loaded {len(df)} samples.")
        else:
            print("[ERROR] No dataset found! Please run setup.sh to download data.")
            sys.exit(1)
            
    # Basic cleaning
    print("   - Shuffling data...")
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def preprocess_data(df):
    print_section("STEP 2: Data Preprocessing")
    print("   - Cleaning text (removing special chars, lowercasing)...")
    # Simple cleaning for demo speed (you can expand this)
    # df['text'] = df['text'].apply(clean_function) 
    
    print("   - Splitting features (Text) and targets (Labels)...")
    X = df['text'] if 'text' in df.columns else df['news_text']
    y = df['label']
    
    return X, y

def train(X, y):
    print_section("STEP 3: Model Training")
    
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    
    # Split
    print("   - Splitting data into Train (80%) and Test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize
    print("   - Vectorizing text data (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features for speed
    X_train_vec = vectorizer.fit_transform(X_train.astype('U'))
    X_test_vec = vectorizer.transform(X_test.astype('U'))
    
    # Train
    print("   - Training Logistic Regression Model...")
    model = LogisticRegression()
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    print("   - Evaluating Model...")
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Model Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer

def save_artifacts(model, vectorizer):
    print_section("STEP 4: Saving Artifacts")
    print(f"   - Saving Model to {MODEL_PATH}")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"   - Saving Vectorizer to {VECTORIZER_PATH}")
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("\n[DONE] Training Complete. Ready for deployment!")

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess_data(df)
    model, vectorizer = train(X, y)
    save_artifacts(model, vectorizer)
