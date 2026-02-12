"""
Enhanced Stress Reduction System
Analyzes psychological impact of news content and provides personalized resources
"""

import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class StressAnalyzer:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        
        # Comprehensive word lists for stress detection
        self.violence_words = [
            'kill', 'death', 'murder', 'attack', 'assault', 'shoot', 'shooting',
            'bomb', 'bombing', 'terror', 'terrorist', 'war', 'weapon', 'gun',
            'victim', 'massacre', 'blood', 'wounded', 'injury', 'dead', 'die',
            'stabbing', 'violent', 'brutally', 'slaughter', 'execute'
        ]
        
        self.fear_words = [
            'shocking', 'panic', 'fear', 'scary', 'terrifying', 'horrifying',
            'nightmare', 'threat', 'danger', 'crisis', 'emergency', 'alarm',
            'warning', 'beware', 'catastrophe', 'disaster', 'apocalypse',
            'collapse', 'chaos', 'outbreak', 'epidemic', 'deadly'
        ]
        
        self.urgency_words = [
            'breaking', 'urgent', 'immediately', 'now', 'hurry', 'quick',
            'before', 'deleted', 'banned', 'censored', 'hidden', 'exposed',
            'must', 'need', 'critical', 'alert', 'imminent', 'emergency'
        ]
        
        self.threat_words = [
            'you', 'your', 'family', 'home', 'children', 'safety', 'health',
            'life', 'risk', 'danger', 'threat', 'target', 'victim'
        ]
        
        self.manipulation_words = [
            'secret', 'conspiracy', 'cover-up', 'hidden', 'exposed', 'truth',
            'they', 'government', 'control', 'lie', 'fake', 'hoax', 'scam',
            'wake up', 'sheeple', 'mainstream media', 'big pharma'
        ]
    
    def analyze_text(self, text, is_fake, confidence):
        """
        Comprehensive stress analysis of text content
        
        Returns:
            dict: Complete stress analysis with score, factors, and resources
        """
        text_lower = text.lower()
        
        # Calculate individual stress factors (0-1 scale)
        violence_score = self._calculate_word_presence(text_lower, self.violence_words)
        fear_score = self._calculate_word_presence(text_lower, self.fear_words)
        urgency_score = self._calculate_word_presence(text_lower, self.urgency_words)
        threat_score = self._calculate_word_presence(text_lower, self.threat_words)
        manipulation_score = self._calculate_word_presence(text_lower, self.manipulation_words)
        
        # Get sentiment for additional context
        sentiment = self.vader.polarity_scores(text)
        negativity_score = abs(sentiment['neg'])  # How negative the content is
        
        # Misinformation anxiety - if it's fake news, adds to stress
        misinformation_score = confidence if is_fake else 0
        
        # Weighted stress calculation (0-100 scale)
        stress_score = (
            violence_score * 25 +      # Violence is highly stressful
            fear_score * 20 +           # Fear-inducing content
            threat_score * 20 +         # Personal threat perception
            urgency_score * 10 +        # Urgency pressure
            manipulation_score * 10 +   # Manipulation tactics
            negativity_score * 10 +     # Overall negativity
            misinformation_score * 5    # Fake news anxiety
        )
        
        # Cap at 100
        stress_score = min(100, stress_score)
        
        # Categorize stress level
        if stress_score >= 70:
            stress_level = 'CRITICAL'
            stress_color = '#dc2626'  # Red
        elif stress_score >= 50:
            stress_level = 'HIGH'
            stress_color = '#f59e0b'  # Orange
        elif stress_score >= 30:
            stress_level = 'MEDIUM'
            stress_color = '#fbbf24'  # Yellow
        else:
            stress_level = 'LOW'
            stress_color = '#22c55e'  # Green
        
        # Identify specific triggers found
        triggers_found = self._identify_triggers(text_lower)
        
        # Get personalized resources
        resources = self._get_resources(stress_level, triggers_found)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(stress_score, triggers_found, is_fake)
        
        return {
            'stress_score': round(stress_score, 1),
            'stress_level': stress_level,
            'stress_color': stress_color,
            'risk_factors': {
                'violence': round(violence_score, 2),
                'fear': round(fear_score, 2),
                'threat': round(threat_score, 2),
                'urgency': round(urgency_score, 2),
                'manipulation': round(manipulation_score, 2),
                'negativity': round(negativity_score, 2),
                'misinformation': round(misinformation_score, 2)
            },
            'triggers_found': triggers_found,
            'resources': resources,
            'recommendations': recommendations
        }
    
    def _calculate_word_presence(self, text, word_list):
        """Calculate presence of words from list in text (0-1 scale)"""
        matches = sum(1 for word in word_list if word in text)
        # Normalize by list size and apply logarithmic scaling
        if matches == 0:
            return 0.0
        # More matches = higher score, but with diminishing returns
        return min(1.0, (matches / len(word_list)) * 10)
    
    def _identify_triggers(self, text):
        """Identify which specific trigger words were found"""
        triggers = {
            'violence': [],
            'fear': [],
            'urgency': [],
            'threat': [],
            'manipulation': []
        }
        
        for word in self.violence_words:
            if word in text:
                triggers['violence'].append(word)
        
        for word in self.fear_words:
            if word in text:
                triggers['fear'].append(word)
        
        for word in self.urgency_words:
            if word in text:
                triggers['urgency'].append(word)
        
        for word in self.threat_words:
            if word in text:
                triggers['threat'].append(word)
        
        for word in self.manipulation_words:
            if word in text:
                triggers['manipulation'].append(word)
        
        # Remove empty categories and limit to top 5 per category
        return {k: v[:5] for k, v in triggers.items() if v}
    
    def _get_resources(self, stress_level, triggers):
        """Get personalized mental health resources based on stress level"""
        resources = []
        
        if stress_level == 'CRITICAL':
            resources.extend([
                {
                    'title': '🆘 National Mental Health Crisis Line',
                    'contact': '988 (call or text)',
                    'description': '24/7 crisis support - immediate help available',
                    'type': 'crisis',
                    'priority': 'URGENT'
                },
                {
                    'title': '💨 Emergency 5-Minute Calm Down',
                    'link': 'https://www.headspace.com/meditation/breathing-exercises',
                    'description': 'Box breathing technique to reduce acute stress',
                    'type': 'breathing',
                    'duration': '5 min'
                },
                {
                    'title': '🚫 Recommended Digital Detox',
                    'description': 'Take a 24-hour break from news and social media',
                    'type': 'advice',
                    'action': 'Step away from screens for at least 1 hour'
                }
            ])
        
        if stress_level in ['CRITICAL', 'HIGH']:
            resources.extend([
                {
                    'title': '🧘 Guided Meditation for Anxiety',
                    'link': 'https://www.calm.com/blog/meditation-for-anxiety',
                    'description': '10-minute guided session to reduce anxiety',
                    'type': 'meditation',
                    'duration': '10 min'
                },
                {
                    'title': '💬 Online Counseling (BetterHelp)',
                    'link': 'https://www.betterhelp.com',
                    'description': 'Professional therapy available today',
                    'type': 'therapy',
                    'cost': 'Varies by plan'
                },
                {
                    'title': '📱 Mental Health Apps',
                    'description': 'Headspace, Calm, or Sanvello for ongoing support',
                    'type': 'app',
                    'cost': 'Free trials available'
                }
            ])
        
        if stress_level in ['MEDIUM', 'HIGH']:
            resources.extend([
                {
                    'title': '🎯 Mindfulness Exercises',
                    'link': 'https://www.mindful.org/meditation/mindfulness-getting-started',
                    'description': 'Simple techniques to stay grounded',
                    'type': 'prevention',
                    'duration': '5-15 min'
                },
                {
                    'title': '📰 News Consumption Tips',
                    'description': 'Limit news to 30 min/day, choose trusted sources',
                    'type': 'advice',
                    'action': 'Set app time limits'
                }
            ])
        
        if stress_level == 'LOW':
            resources.extend([
                {
                    'title': '✅ Good News Practice',
                    'description': 'Balance negative news with positive stories',
                    'type': 'prevention',
                    'action': 'Read one positive news story today'
                },
                {
                    'title': '🧠 Mental Health Maintenance',
                    'link': 'https://www.mentalhealth.gov/basics/what-is-mental-health',
                    'description': 'Learn about maintaining good mental health',
                    'type': 'education'
                }
            ])
        
        return resources
    
    def _generate_recommendations(self, stress_score, triggers, is_fake):
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if stress_score >= 70:
            recommendations.append({
                'priority': 'URGENT',
                'icon': '🚨',
                'text': 'This content shows high stress indicators. Consider taking a break from news consumption.',
                'action': 'Take at least 30 minutes away from screens'
            })
        
        if is_fake and stress_score >= 50:
            recommendations.append({
                'priority': 'HIGH',
                'icon': '⚠️',
                'text': 'This appears to be misinformation designed to provoke emotional response.',
                'action': 'Verify through trusted fact-checking sources before sharing'
            })
        
        if 'violence' in triggers and len(triggers['violence']) > 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'icon': '💔',
                'text': 'Content contains graphic/violent themes that may be disturbing.',
                'action': 'Practice grounding techniques if feeling overwhelmed'
            })
        
        if 'manipulation' in triggers:
            recommendations.append({
                'priority': 'MEDIUM',
                'icon': '🎭',
                'text': 'Content uses emotional manipulation tactics common in misinformation.',
                'action': 'Approach with skepticism and verify claims independently'
            })
        
        if stress_score >= 30:
            recommendations.append({
                'priority': 'LOW',
                'icon': '🧘',
                'text': 'Consider a brief mindfulness exercise after reading stressful content.',
                'action': 'Try 5 minutes of deep breathing or meditation'
            })
        
        return recommendations

# Singleton instance
stress_analyzer = StressAnalyzer()
