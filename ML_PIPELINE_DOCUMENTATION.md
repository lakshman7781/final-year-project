# TruthGuard: Fake News Detection - ML Pipeline Documentation

## Project Overview

**TruthGuard** is an AI-powered fake news detection system that uses Natural Language Processing (NLP) and Machine Learning to classify news articles as real or fake with high accuracy.

### Tech Stack
- **Language**: Python 3.12
- **ML Framework**: scikit-learn
- **NLP**: NLTK, TF-IDF Vectorization
- **Algorithm**: Logistic Regression
- **Backend**: Flask
- **Model Storage**: Pickle serialization

---

## 1. Dataset

### Source
**ISOT Fake News Dataset** from Kaggle
- Dataset: `clmentbisaillon/fake-and-real-news-dataset`
- Files: `True.csv` (Real News), `Fake.csv` (Fake News)

### Dataset Statistics
| Metric | Value |
|--------|-------|
| **Total Articles** | 44,898 |
| **Real Articles** | ~21,417 |
| **Fake Articles** | ~23,481 |
| **Average Article Length** | 2,383 characters (Real), 2,547 characters (Fake) |
| **File Size** | 112 MB (60MB Fake + 52MB Real) |

### Dataset Structure
```
Columns:
- title: Article headline
- text: Full article content
- subject: Topic category
- date: Publication date
```

---

## 2. Data Pipeline

### 2.1 Data Loading
**File**: `ml_engine/train_model.py` - `load_data()`

```python
Process:
1. Load True.csv → Label as 1 (Real)
2. Load Fake.csv → Label as 0 (Fake)
3. Concatenate both datasets
4. Shuffle data randomly (random_state for reproducibility)
```

**Key Features**:
- Automatic dataset detection from `data/` directory
- Fallback to small sample dataset if main data missing
- Data shuffling to prevent order bias

### 2.2 Data Preprocessing
**File**: `ml_engine/train_model.py` - `preprocess_data()`

**Text Cleaning Function**:
```python
def clean_text(text):
    text = str(text).lower()                    # Lowercase normalization
    text = re.sub(r'\[.*?\]', '', text)        # Remove [text] patterns
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)         # Remove HTML tags
    text = re.sub(r'\n', '', text)             # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)       # Remove words with numbers
    return text
```

**Preprocessing Steps**:
1. **Text Extraction**: Extract 'text' column as features
2. **Label Extraction**: Extract 'label' column as target (0=Fake, 1=Real)
3. **Cleaning**: Apply text cleaning transformations
4. **Normalization**: Convert to lowercase, remove noise

### 2.3 Train-Test Split
```python
Split Ratio: 80% Training / 20% Testing
Method: train_test_split(X, y, test_size=0.2, random_state=42)

Training Set: 35,918 articles
Testing Set:   8,980 articles
```

---

## 3. Feature Engineering

### TF-IDF Vectorization
**Technique**: Term Frequency-Inverse Document Frequency

```python
Vectorizer: TfidfVectorizer(max_features=5000)
```

**Configuration**:
- **max_features**: 5,000 (top 5000 most important words)
- **Output**: Sparse matrix of shape (n_samples, 5000)

**How TF-IDF Works**:
1. **Term Frequency (TF)**: Measures how frequently a term occurs in a document
   - `TF = (Number of times term t appears in document) / (Total terms in document)`

2. **Inverse Document Frequency (IDF)**: Measures importance of term across all documents
   - `IDF = log(Total number of documents / Number of documents containing term t)`

3. **TF-IDF Score**: `TF × IDF`
   - High score: Word is frequent in document but rare across corpus (important)
   - Low score: Common word appearing everywhere (less important)

**Why TF-IDF?**
- Captures semantic importance of words
- Reduces impact of common words (the, is, are)
- Creates numerical representation of text
- Efficient for large datasets

---

## 4. Model Architecture

### Algorithm: Logistic Regression

**Choice Rationale**:
- ✅ Fast training on large datasets
- ✅ Interpretable results (feature coefficients)
- ✅ Probability outputs (confidence scores)
- ✅ Works well with TF-IDF features
- ✅ Low memory footprint
- ✅ Proven effectiveness for text classification

### Model Configuration
```python
Model: LogisticRegression()
Solver: lbfgs (default)
Max Iterations: 100 (default)
Penalty: l2 (Ridge regularization)
```

### Training Process
**File**: `ml_engine/train_model.py` - `train()`

```
Step 1: Fit TF-IDF Vectorizer on training text
        → Learn vocabulary and IDF weights

Step 2: Transform training text to TF-IDF matrix
        → Convert text to numerical features

Step 3: Train Logistic Regression classifier
        → Learn decision boundary between real/fake

Step 4: Transform test text to TF-IDF matrix
        → Apply same vectorization to test data

Step 5: Evaluate model on test set
        → Generate predictions and metrics
```

---

## 5. Model Performance

### Evaluation Metrics

```
Overall Accuracy: 98.90%

Classification Report:
              precision    recall  f1-score   support

        Fake      0.99      0.99      0.99      4,744
        Real      0.99      0.99      0.99      4,236

    accuracy                          0.99      8,980
   macro avg      0.99      0.99      0.99      8,980
weighted avg      0.99      0.99      0.99      8,980
```

### Metric Definitions

**Precision**: Of all articles predicted as fake, how many were actually fake?
- `Precision = True Positives / (True Positives + False Positives)`
- **Our Score**: 99% (extremely reliable when flagging fake news)

**Recall**: Of all actual fake articles, how many did we correctly identify?
- `Recall = True Positives / (True Positives + False Negatives)`
- **Our Score**: 99% (catches almost all fake news)

**F1-Score**: Harmonic mean of Precision and Recall
- `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- **Our Score**: 99% (balanced performance)

**Support**: Number of actual occurrences in test set

### Confusion Matrix (Estimated)
```
                 Predicted Fake  Predicted Real
Actual Fake           4,696            48
Actual Real              51          4,185
```

---

## 6. Model Serialization & Deployment

### Saved Artifacts
**File**: `ml_engine/train_model.py` - `save_artifacts()`

```python
1. fake_news_model.pkl    → Trained Logistic Regression model
2. vectorizer.pkl         → Fitted TF-IDF Vectorizer
```

**Why Save Both?**
- Model needs same vocabulary/features seen during training
- Vectorizer ensures consistent feature extraction at inference

### Loading for Inference
**File**: `ml_engine/predictor.py` - `FakeNewsPredictor`

```python
class FakeNewsPredictor:
    Load saved model and vectorizer
    ↓
    Transform new text using vectorizer
    ↓
    Predict using model
    ↓
    Return prediction + confidence score
```

---

## 7. Prediction Pipeline

### Input Validation
**File**: `ml_engine/predictor.py` - `predict()`

```python
Input Validation Rules:
1. Text length check:
   - If < 100 characters or < 15 words:
     → Return warning (model trained on full articles)
   - If ≥ 100 characters:
     → Proceed with prediction

2. Text preprocessing:
   - Apply same cleaning as training data
   - Ensure consistency
```

### Prediction Flow
```
User Input Text
    ↓
Input Validation (length check)
    ↓
Transform to TF-IDF (using saved vectorizer)
    ↓
Model Prediction (Logistic Regression)
    ↓
Get Probability Scores [P(Fake), P(Real)]
    ↓
Extract Confidence (max probability)
    ↓
Return {is_fake, confidence, stress_level}
```

### Output Format
```json
{
    "is_fake": false,
    "confidence": 0.9890,
    "stress_level": "Low",
    "warning": "Optional warning for short inputs",
    "suggestion": "Optional suggestion for improvement"
}
```

---

## 8. API Integration

### Flask Backend
**File**: `backend/app.py`

**Endpoint**: `POST /api/analyze`

```python
Request Body:
{
    "text": "News article or headline to analyze"
}

Response:
{
    "is_fake": boolean,
    "confidence": float (0-1),
    "stress_level": "High" | "Low",
    "sentiment_score": float,
    "sentiment_label": "Positive" | "Negative" | "Neutral",
    "triggers": ["sensational", "words"],
    "fact_checks": [...]
}
```

### Additional Features

**1. Sentiment Analysis** (TextBlob)
- Polarity score: -1 (negative) to +1 (positive)
- Subjectivity: 0 (objective) to 1 (subjective)

**2. Trigger Word Detection**
- Detects sensational words: "shocking", "banned", "conspiracy"
- Helps identify emotional manipulation

**3. Live Fact-Checking** (Google News Scraper)
- Searches related articles on Google News
- Provides source verification links

---

## 9. Model Limitations & Considerations

### Current Limitations

1. **Input Length Sensitivity**
   - Trained on full articles (~2,400 characters)
   - Short statements may not provide enough features
   - **Solution**: Input validation with user warnings

2. **Domain Specificity**
   - Trained on 2022 dataset
   - May not recognize newer fake news patterns
   - **Solution**: Regular retraining with new data

3. **Language Support**
   - English-only model
   - Cannot handle multilingual content
   - **Future**: Multi-language support

4. **Context Understanding**
   - Relies on word patterns, not semantic understanding
   - Cannot verify factual claims
   - **Mitigation**: Integrated fact-checking API

### Strengths

✅ **High Accuracy**: 98.90% on test set
✅ **Fast Inference**: Real-time predictions
✅ **Interpretable**: Can analyze feature weights
✅ **Scalable**: Efficient with large datasets
✅ **Low Resource**: Doesn't require GPU

---

## 10. Future Improvements

### Short-term
1. ✅ Add input validation (completed)
2. ✅ Improve UI warnings (completed)
3. Add more trigger words
4. Enhance fact-checking coverage

### Medium-term
1. Implement ensemble models (combine multiple algorithms)
2. Add BERT/Transformer models for better context
3. Real-time learning from user feedback
4. Credibility scoring system

### Long-term
1. Multi-language support
2. Image/video fake news detection
3. Social media integration
4. Browser extension for live fact-checking

---

## 11. Technical Requirements

### Dependencies
```
flask              → Web framework
scikit-learn       → ML algorithms
pandas             → Data manipulation
numpy              → Numerical operations
nltk               → NLP utilities
joblib/pickle      → Model serialization
textblob           → Sentiment analysis
beautifulsoup4     → Web scraping
requests           → HTTP client
```

### System Requirements
- **Python**: 3.8+
- **RAM**: 2GB minimum (4GB recommended)
- **Storage**: 500MB for models and data
- **CPU**: Any modern processor (no GPU required)

---

## 12. Training Time & Resources

### Training Performance
```
Dataset Size:      44,898 articles (112 MB)
Training Time:     ~2-3 minutes on standard CPU
TF-IDF Fitting:    ~30 seconds
Model Training:    ~60 seconds
Evaluation:        ~10 seconds
```

### Model Size
```
fake_news_model.pkl:  ~1.5 MB
vectorizer.pkl:       ~3.5 MB
Total:                ~5 MB
```

---

## 13. Reproducibility

### Random Seeds
```python
train_test_split(random_state=42)  → Consistent data splits
df.sample(frac=1, random_state=42) → Reproducible shuffling
```

### Environment
```bash
Virtual Environment: Python 3.12
Package Manager: pip
Requirements: requirements.txt (pinned versions)
```

---

## Conclusion

TruthGuard demonstrates a complete end-to-end ML pipeline for fake news detection:

1. ✅ **Data Collection**: 44K+ articles from reliable sources
2. ✅ **Preprocessing**: Text cleaning and normalization
3. ✅ **Feature Engineering**: TF-IDF vectorization
4. ✅ **Model Training**: Logistic Regression with 98.90% accuracy
5. ✅ **Deployment**: Flask API with real-time predictions
6. ✅ **User Experience**: Web interface with fact-checking

The system balances **accuracy**, **speed**, and **interpretability** to provide a practical solution for combating misinformation.

---

**Project Repository**: https://github.com/lakshman7781/final-year-project

**Author**: lakshman7781

**Last Updated**: February 12, 2026
