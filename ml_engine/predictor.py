import joblib
import os
import pickle
import numpy as np
import re

class FakeNewsPredictor:
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, 'fake_news_model.pkl')
        self.vectorizer_path = os.path.join(self.base_dir, 'vectorizer.pkl')
        
        self.model = None
        self.vectorizer = None
        self.load_model()

    def load_model(self):
        try:
            print(f"Loading model from {self.model_path}...")
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("Model loaded successfully.")
            else:
                print(f"Error: Model file at {self.model_path} not found.")

            print(f"Loading vectorizer from {self.vectorizer_path}...")
            if os.path.exists(self.vectorizer_path):
                 with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                 print("Vectorizer loaded successfully.")
            else:
                 print(f"Error: Vectorizer file at {self.vectorizer_path} not found.")
                 
        except Exception as e:
            print(f"Error loading model/vectorizer: {e}")

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>+', '', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w*\d\w*', '', text)
        return text

    def predict(self, text):
        if not self.model or not self.vectorizer:
            return {"error": "Model or Vectorizer not loaded"}
        
        try:
            # Check text length - model trained on articles ~2400 chars
            word_count = len(text.split())
            char_count = len(text)
            
            # Handle very short inputs (statements vs articles)
            if char_count < 100 or word_count < 15:
                return {
                    "is_fake": False,
                    "confidence": 0.5,
                    "stress_level": "Low",
                    "warning": "Input too short for reliable analysis. Model trained on full articles (avg. 2400 characters). Please provide more context or a complete article for accurate detection.",
                    "suggestion": "Try entering a full news article or headline with supporting details."
                }
            
            # 1. Transform text
            # Note: We rely on the vectorizer's internal preprocessing
            text_vec = self.vectorizer.transform([text])
            
            # 2. Predict
            prediction = self.model.predict(text_vec)[0]
            proba = self.model.predict_proba(text_vec)[0]
            
            # 3. Format result
            confidence = float(np.max(proba))
            is_fake = bool(prediction == 0) # 0 = Fake, 1 = Real
            
            return {
                "is_fake": is_fake,
                "confidence": confidence,
                "stress_level": "High" if is_fake and confidence > 0.8 else "Low"
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {"error": str(e)}
