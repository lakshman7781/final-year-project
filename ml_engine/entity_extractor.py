"""
Entity Extraction Module
Uses spaCy NER to extract people, organizations, locations from text
"""

import spacy

class EntityExtractor:
    def __init__(self):
        """Initialize spaCy model for Named Entity Recognition"""
        try:
            self.nlp = spacy.load('en_core_web_sm')
            print("Entity extractor loaded successfully.")
        except Exception as e:
            print(f"Error loading spaCy model: {e}")
            self.nlp = None
    
    def extract_entities(self, text):
        """
        Extract named entities from text
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            dict: Categorized entities with counts and examples
        """
        if not self.nlp:
            return {'error': 'Entity extraction not available'}
        
        try:
            doc = self.nlp(text)
            
            # Categorize entities
            entities = {
                'PERSON': [],       # People
                'ORG': [],          # Organizations, companies
                'GPE': [],          # Countries, cities, states
                'PRODUCT': [],      # Products, brands
                'EVENT': [],        # Named events
                'LAW': [],          # Laws, legal documents
                'FAC': [],          # Buildings, facilities
                'NORP': []          # Nationalities, religious/political groups
            }
            
            # Extract unique entities
            seen = set()
            for ent in doc.ents:
                if ent.label_ in entities:
                    # Avoid duplicates (case-insensitive)
                    if ent.text.lower() not in seen:
                        entities[ent.label_].append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char
                        })
                        seen.add(ent.text.lower())
            
            # Remove empty categories
            entities = {k: v for k, v in entities.items() if v}
            
            # Create summary
            total_entities = sum(len(v) for v in entities.values())
            
            # Get top entities by category
            summary = {
                'total_count': total_entities,
                'categories': list(entities.keys()),
                'entities_by_category': {
                    k: [e['text'] for e in v[:5]]  # Top 5 per category
                    for k, v in entities.items()
                },
                'all_entities': entities,
                'primary_subjects': self._identify_primary_subjects(entities)
            }
            
            return summary
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return {'error': str(e)}
    
    def _identify_primary_subjects(self, entities):
        """
        Identify the main people/organizations mentioned
        (Most likely reputation management targets)
        """
        primary = []
        
        # Prioritize people and organizations as they're most relevant
        # for reputation management
        if 'PERSON' in entities and entities['PERSON']:
            primary.extend([
                {
                    'name': e['text'],
                    'type': 'Person',
                    'icon': '👤'
                }
                for e in entities['PERSON'][:3]  # Top 3
            ])
        
        if 'ORG' in entities and entities['ORG']:
            primary.extend([
                {
                    'name': e['text'],
                    'type': 'Organization',
                    'icon': '🏢'
                }
                for e in entities['ORG'][:3]  # Top 3
            ])
        
        # Limit to top 5 primary subjects
        return primary[:5]

# Singleton instance
entity_extractor = EntityExtractor()
