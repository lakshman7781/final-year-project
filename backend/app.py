from flask import Flask, request, jsonify, render_template
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_engine.predictor import FakeNewsPredictor
from backend.scraper import GoogleNewsScraper
from ml_engine.stress_analyzer import stress_analyzer
from ml_engine.entity_extractor import entity_extractor

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
                "stress_level": result.get('stress_level', 'Unknown'),
                "stress_score": result.get('stress_score', 0),
                "entities": result.get('entities', {})
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
    
    # NEW: Enhanced Stress Analysis
    stress_analysis = stress_analyzer.analyze_text(
        text,
        result['is_fake'],
        result['confidence']
    )
    result['stress_analysis'] = stress_analysis
    result['stress_score'] = stress_analysis['stress_score']
    result['stress_level'] = stress_analysis['stress_level']
    
    # NEW: Entity Extraction
    entities = entity_extractor.extract_entities(text)
    result['entities'] = entities
    
    # Log to history
    history_manager.log(text, result)
    
    return jsonify(result)

@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(history_manager.get_history())

@app.route('/api/stress-relief', methods=['GET'])
def stress_relief():
    """
    Get stress relief resources based on most recent analysis
    """
    try:
        history = history_manager.get_history()
        if history and 'stress_analysis' in history[-1]:
            stress_data = history[-1].get('stress_analysis', {})
            resources = stress_data.get('resources', [])
            return jsonify({
                'resources': resources,
                'stress_score': stress_data.get('stress_score', 0),
                'stress_level': stress_data.get('stress_level', 'Unknown'),
                'recommendations': stress_data.get('recommendations', [])
            })
    except Exception as e:
        print(f"Error getting stress relief: {e}")
    
    # Fallback to default resources
    resources = [
        {
            'title': 'Mental Health Helpline',
            'contact': '988',
            'description': '24/7 crisis support',
            'type': 'crisis'
        },
        {
            'title': 'Guided Meditation',
            'link': 'https://www.headspace.com/meditation',
            'description': '10-minute stress relief',
            'type': 'meditation'
        },
        {
            'title': 'Digital Detox Tips',
            'description': 'Limit news to 30 min/day',
            'type': 'advice'
        }
    ]
    return jsonify({'resources': resources})

@app.route('/api/reputation', methods=['GET'])
def reputation():
    """
    Get reputation summary for entities mentioned in last analyzed text
    """
    history = history_manager.get_history()
    
    if not history:
        return jsonify({
            "entity": "No analysis yet",
            "score": 50,
            "mentions": [],
            "action_items": ["Analyze some news to see reputation insights"]
        })
    
    last_entry = history[-1]
    entities_data = last_entry.get('entities', {})
    
    # Get primary subjects (people/organizations)
    primary_subjects = entities_data.get('primary_subjects', [])
    
    if primary_subjects:
        entity_name = primary_subjects[0]['name']
        entity_type = primary_subjects[0]['type']
    else:
        # Fallback to trying to extract from text
        try:
            text_source = last_entry.get('full_text', last_entry.get('text_preview', ''))
            blob = TextBlob(text_source)
            if blob.noun_phrases:
                entity_name = blob.noun_phrases[0].title()
                entity_type = "Entity"
            else:
                entity_name = "Subject"
                entity_type = "Unknown"
        except:
            entity_name = "Subject"
            entity_type = "Unknown"
    
    # Calculate reputation factors
    is_fake = last_entry.get('is_fake', False)
    confidence = last_entry.get('confidence', 0.5)
    sentiment_score = last_entry.get('sentiment_score', 0)
    stress_score = last_entry.get('stress_score', 0)
    
    # Reputation scoring
    base_score = 75  # Neutral starting point
    
    # Adjust based on authenticity
    if is_fake:
        base_score -= (confidence * 30)  # Fake news hurts reputation
    
    # Adjust based on sentiment
    base_score += (sentiment_score * 10)  # Positive sentiment helps
    
    # Adjust based on stress/negativity
    base_score -= (stress_score * 0.2)  # High stress content hurts
    
    # Cap between 0-100
    reputation_score = max(0, min(100, base_score))
    
    # Generate mentions summary
    mentions = []
    if is_fake:
        mentions.append({
            "source": "Fake News Detection",
            "sentiment": "Negative",
            "text": f"{entity_name} mentioned in detected fake news article",
            "impact": "High negative impact on reputation"
        })
    else:
        if sentiment_score > 0.1:
            mentions.append({
                "source": "News Analysis",
                "sentiment": "Positive",
                "text": f"{entity_name} mentioned in positive context",
                "impact": "Neutral to positive impact"
            })
        elif sentiment_score < -0.1:
            mentions.append({
                "source": "News Analysis",
                "sentiment": "Negative",
                "text": f"{entity_name} mentioned in negative context",
                "impact": "Moderate negative impact"
            })
        else:
            mentions.append({
                "source": "News Analysis",
                "sentiment": "Neutral",
                "text": f"{entity_name} mentioned in neutral reporting",
                "impact": "Minimal impact"
            })
    
    # Generate action items
    action_items = []
    if is_fake and confidence > 0.7:
        action_items.append(f"🚨 URGENT: Submit fact-check request for claims about {entity_name}")
        action_items.append(f"Consider issuing official statement clarifying misinformation")
    elif sentiment_score < -0.3:
        action_items.append(f"Monitor negative sentiment trends for {entity_name}")
        action_items.append(f"Consider engaging with community to address concerns")
    elif reputation_score >= 70:
        action_items.append(f"Reputation is healthy - maintain current communication strategy")
    else:
        action_items.append(f"Review recent mentions and address any concerns")
    
    # Add all detected entities as context
    all_entities_summary = []
    if 'entities_by_category' in entities_data:
        for category, entity_list in entities_data['entities_by_category'].items():
            all_entities_summary.extend([
                f"{category}: {', '.join(entity_list[:3])}"
            ])
    
    report = {
        "entity": entity_name,
        "entity_type": entity_type,
        "score": round(reputation_score, 1),
        "grade": _get_reputation_grade(reputation_score),
        "mentions": mentions,
        "action_items": action_items,
        "all_entities": all_entities_summary,
        "total_entities_detected": entities_data.get('total_count', 0)
    }
    return jsonify(report)

def _get_reputation_grade(score):
    """Convert reputation score to letter grade"""
    if score >= 90:
        return "A (Excellent)"
    elif score >= 80:
        return "B (Good)"
    elif score >= 70:
        return "C (Fair)"
    elif score >= 60:
        return "D (Poor)"
    else:
        return "F (Critical)"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
