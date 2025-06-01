from flask import Flask, request, jsonify
import logging
import os
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components with error handling
pii_masker = None
email_classifier = None
model_loaded = False

def initialize_components():
    """Initialize PII masker and email classifier with error handling"""
    global pii_masker, email_classifier, model_loaded
    
    try:
        # Import and initialize PIIMasker
        from utils import PIIMasker
        pii_masker = PIIMasker()
        logger.info("PIIMasker initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PIIMasker: {str(e)}")
        logger.error(traceback.format_exc())
    
    try:
        # Import and initialize EmailClassifierTrainer
        from models import EmailClassifierTrainer
        email_classifier = EmailClassifierTrainer()
        logger.info("EmailClassifierTrainer initialized successfully")
        
        # Try to load pre-trained model
        model_paths = [
            'models/email_classifier.pkl',
            'email_classifier.pkl',
            './models/email_classifier.pkl'
        ]
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    email_classifier.load_model(model_path)
                    logger.info(f"Email classifier model loaded successfully from {model_path}")
                    model_loaded = True
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model from {model_path}: {str(e)}")
        
        if not model_loaded:
            logger.warning("No pre-trained model found. API will use fallback classification.")
            
    except Exception as e:
        logger.error(f"Failed to initialize EmailClassifierTrainer: {str(e)}")
        logger.error(traceback.format_exc())

# Initialize components
initialize_components()

def fallback_classification(email_text):
    """Simple fallback classification based on keywords"""
    try:
        email_lower = email_text.lower()
        
        # Define keyword-based rules
        if any(word in email_lower for word in ['bill', 'billing', 'payment', 'invoice', 'charge', 'refund']):
            return "Request", 0.7
        elif any(word in email_lower for word in ['technical', 'error', 'bug', 'not working', 'issue', 'problem']):
            return "Problem", 0.7
        elif any(word in email_lower for word in ['account', 'login', 'password', 'access']):
            return "Request", 0.7
        elif any(word in email_lower for word in ['change', 'update', 'modify', 'upgrade']):
            return "Change", 0.7
        elif any(word in email_lower for word in ['incident', 'outage', 'down', 'urgent']):
            return "Incident", 0.7
        else:
            return "Request", 0.5
    except Exception as e:
        logger.error(f"Fallback classification error: {str(e)}")
        return "Request", 0.5

def fallback_pii_masking(email_text):
    """Simple fallback PII masking using regex"""
    try:
        import re
        
        entities = []
        masked_email = email_text
        
        # Simple email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, email_text):
            entities.append({
                'position': [match.start(), match.end()],
                'classification': 'email',
                'entity': match.group()
            })
            masked_email = masked_email.replace(match.group(), '[email]')
        
        # Simple phone pattern
        phone_pattern = r'(?:\+91[-.\s]?)?(?:\d{10}|\d{3}[-.\s]?\d{3}[-.\s]?\d{4})'
        for match in re.finditer(phone_pattern, email_text):
            entities.append({
                'position': [match.start(), match.end()],
                'classification': 'phone_number',
                'entity': match.group()
            })
            masked_email = masked_email.replace(match.group(), '[phone_number]')
        
        return entities, masked_email
        
    except Exception as e:
        logger.error(f"Fallback PII masking error: {str(e)}")
        return [], email_text

@app.route('/classify', methods=['POST'])
def classify_email():
    """
    Classify an email and mask PII entities
    """
    try:
        # Get input data
        data = request.get_json()
        
        if not data or 'input_email_body' not in data:
            return jsonify({
                'error': 'Missing required field: input_email_body'
            }), 400
        
        input_email_body = data['input_email_body']
        
        if not isinstance(input_email_body, str):
            return jsonify({
                'error': 'input_email_body must be a string'
            }), 400
        
        if not input_email_body.strip():
            return jsonify({
                'error': 'input_email_body cannot be empty'
            }), 400
        
        # Process PII masking
        logger.info("Processing PII masking...")
        try:
            if pii_masker:
                masked_entities, masked_email = pii_masker.process_email(input_email_body)
            else:
                logger.warning("Using fallback PII masking")
                masked_entities, masked_email = fallback_pii_masking(input_email_body)
        except Exception as e:
            logger.error(f"PII masking error: {str(e)}")
            logger.warning("Using fallback PII masking")
            masked_entities, masked_email = fallback_pii_masking(input_email_body)
        
        # Classify the email
        logger.info("Classifying email...")
        try:
            if model_loaded and email_classifier:
                category, confidence = email_classifier.predict(input_email_body)
                logger.info(f"ML Classification: {category} (confidence: {confidence:.3f})")
            else:
                category, confidence = fallback_classification(input_email_body)
                logger.info(f"Fallback Classification: {category} (confidence: {confidence:.3f})")
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            category, confidence = fallback_classification(input_email_body)
            logger.info(f"Error fallback Classification: {category} (confidence: {confidence:.3f})")
        
        # Prepare response
        response = {
            "input_email_body": input_email_body,
            "list_of_masked_entities": masked_entities,
            "masked_email": masked_email,
            "category_of_the_email": category
        }
        
        logger.info(f"Successfully processed email. Category: {category}, Entities masked: {len(masked_entities)}")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in classify_email: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        return jsonify({
            'status': 'healthy',
            'message': 'Email classification API is running',
            'model_loaded': model_loaded,
            'pii_masker_available': pii_masker is not None,
            'email_classifier_available': email_classifier is not None
        }), 200
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    try:
        return jsonify({
            'message': 'Email Classification API',
            'endpoints': {
                'classify': '/classify (POST)',
                'health': '/health (GET)'
            },
            'usage': 'Send POST request to /classify with {"input_email_body": "your email text"}',
            'model_status': 'loaded' if model_loaded else 'using fallback classification',
            'pii_masker_status': 'available' if pii_masker else 'using fallback masking'
        }), 200
    except Exception as e:
        logger.error(f"Root endpoint error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

# Add error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)