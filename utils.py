import re
import spacy
from typing import List, Dict, Tuple

class PIIMasker:
    def __init__(self):
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Some features may be limited.")
            self.nlp = None
        
        # Define regex patterns for PII detection
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone_number': r'(?:\+91[-.\s]?)?(?:\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4})',
            'credit_debit_no': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'cvv_no': r'\b\d{3,4}\b(?=.*(?:cvv|cvc|security|code))',
            'expiry_no': r'\b(?:0[1-9]|1[0-2])\/(?:20)?\d{2}\b',
            'aadhar_num': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'dob': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})\b'
        }
    
    def extract_names_with_spacy(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract person names using spaCy NER"""
        names = []
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    names.append((ent.start_char, ent.end_char, ent.text))
        return names
    
    def extract_names_with_regex(self, text: str) -> List[Tuple[int, int, str]]:
        """Extract potential names using regex patterns"""
        # Pattern for capitalized words that might be names
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        names = []
        for match in re.finditer(name_pattern, text):
            # Filter out common non-name phrases
            name_text = match.group().lower()
            exclude_words = ['dear sir', 'dear madam', 'thank you', 'best regards', 
                           'yours sincerely', 'yours faithfully', 'customer service']
            if not any(excluded in name_text for excluded in exclude_words):
                names.append((match.start(), match.end(), match.group()))
        return names
    
    def detect_pii_entities(self, text: str) -> List[Dict]:
        """Detect all PII entities in the text"""
        entities = []
        
        # Detect regex-based entities
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append({
                    'position': [match.start(), match.end()],
                    'classification': entity_type,
                    'entity': match.group()
                })
        
        # Detect names using spaCy and regex
        spacy_names = self.extract_names_with_spacy(text)
        regex_names = self.extract_names_with_regex(text)
        
        # Combine and deduplicate names
        all_names = spacy_names + regex_names
        unique_names = []
        for start, end, name in all_names:
            if not any(abs(start - existing['position'][0]) < 5 for existing in unique_names):
                unique_names.append({
                    'position': [start, end],
                    'classification': 'full_name',
                    'entity': name
                })
        
        entities.extend(unique_names)
        
        # Sort entities by position
        entities.sort(key=lambda x: x['position'][0])
        return entities
    
    def mask_entities(self, text: str, entities: List[Dict]) -> str:
        """Mask entities in the text"""
        masked_text = text
        offset = 0
        
        for entity in entities:
            start_pos = entity['position'][0] + offset
            end_pos = entity['position'][1] + offset
            entity_type = entity['classification']
            
            # Create mask
            mask = f"[{entity_type}]"
            
            # Replace in text
            masked_text = masked_text[:start_pos] + mask + masked_text[end_pos:]
            
            # Update offset
            offset += len(mask) - (end_pos - start_pos)
        
        return masked_text
    
    def process_email(self, email_text: str) -> Tuple[List[Dict], str]:
        """Process email to detect and mask PII entities"""
        entities = self.detect_pii_entities(email_text)
        masked_email = self.mask_entities(email_text, entities)
        
        return entities, masked_email