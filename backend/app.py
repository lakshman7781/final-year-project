from flask import Flask, request, jsonify, render_template
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.predictor import FakeNewsPredictor
from backend.scraper import GoogleNewsScraper

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend', static_url_path='')

import json
from datetime import datetime

class HistoryManager:
    def __init__(self, filepath='history.json'):
        self.filepath = filepath
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                json.dump([], f)

    def log(self, text, result):
        try:
            with open(self.filepath, 'r') as f:
                history = json.load(f)
            
            entry = {
                "timestamp": datetime.now().isoformat(),
                "text_preview": text[:50] + "...",
                "full_text": text,
                "is_fake": result['is_fake'],
                "confidence": result['confidence'],
                "stress_level": result.get('stress_level', 'Unknown')
            }
            history.append(entry)
            
            with open(self.filepath, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Error logging history: {e}")

    def get_history(self):
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except:
            return []

# Initialize components
predictor = FakeNewsPredictor()
scraper = GoogleNewsScraper()
history_manager = HistoryManager()

@app.route('/')
def index():
    return render_template('index.html')

from textblob import TextBlob

# Sensational/Trigger words list
TRIGGER_WORDS = [
    'shocking', 'miracle', 'secret', 'banned', 'illegal', 'death', 'kill',
    'panic', 'terror', 'apocalypse', 'exposed', 'conspiracy', 'dictator',
    'hoax', 'fake', 'destroy', 'explode', 'monster', 'alien', 'zombie'
]

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # ML Prediction
    result = predictor.predict(text)
    
    # Live Fact-Checking
    fact_checks = scraper.search(text)
    result['fact_checks'] = fact_checks
    
    # Advanced Analysis: Sentiment
    blob = TextBlob(text)
    sentiment = blob.sentiment
    result['sentiment_score'] = sentiment.polarity
    result['subjectivity_score'] = sentiment.subjectivity
    
    if sentiment.polarity > 0.1:
        result['sentiment_label'] = "Positive"
    elif sentiment.polarity < -0.1:
        result['sentiment_label'] = "Negative"
    else:
        result['sentiment_label'] = "Neutral"
        
    # Advanced Analysis: Triggers
    found_triggers = [word for word in TRIGGER_WORDS if word in text.lower()]
    result['triggers'] = list(set(found_triggers))
    
    # Log to history
    history_manager.log(text, result)
    
    return jsonify(result)

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(history_manager.get_history())

@app.route('/api/stress-relief', methods=['GET'])
def stress_relief():
    resources = [
        {"title": "Meditation Guide", "link": "#", "desc": "10-minute guided meditation"},
        {"title": "Deep Breathing", "link": "#", "desc": "Breathing exercises for anxiety"},
        {"title": "Professional Help", "link": "#", "desc": "Contact a counselor"}
    ]
    return jsonify(resources)

@app.route('/api/reputation', methods=['GET'])
def reputation():
    # Attempt to get the last analyzed text from history to find the entity
    history = history_manager.get_history()
    entity_name = "User/Organization"
    
    if history:
        last_entry = history[-1]
        try:
            # Use full text if available, otherwise preview
            text_source = last_entry.get('full_text', last_entry.get('text_preview', ''))
            blob = TextBlob(text_source)
            if blob.noun_phrases:
                # Pick the most frequent or first noun phrase
                entity_name = blob.noun_phrases[0].title()
        except:
            pass

    # Mock data for demonstration
    report = {
        "entity": entity_name,
        "score": 85,
        "mentions": [
            {"source": "Twitter", "sentiment": "Negative", "text": f"Fake news detected about {entity_name}..."},
            {"source": "Facebook", "sentiment": "Neutral", "text": f"Mentioned in passing regarding {entity_name}..."}
        ],
        "action_items": [f"Clarify statement about {entity_name}", "Report fake post"]
    }
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
