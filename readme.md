---
title: Email Classifier API
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# Email Classification API

This API classifies support emails into categories (Incident, Request, Change, Problem) and masks personally identifiable information (PII).

## Features

- **Email Classification**: Categorizes emails using machine learning
- **PII Masking**: Detects and masks sensitive information including:
  - Full names
  - Email addresses
  - Phone numbers
  - Credit card numbers
  - CVV numbers
  - Expiry dates
  - Aadhar numbers
  - Date of birth

## API Endpoints

### POST /classify

Classifies an email and masks PII entities.

**Request Body:**
```json
{
  "input_email_body": "Hello, my name is John Doe and my email is john@example.com. I need help with billing."
}
```

**Response:**
```json
{
  "input_email_body": "Original email text",
  "list_of_masked_entities": [
    {
      "position": [17, 25],
      "classification": "full_name",
      "entity": "John Doe"
    },
    {
      "position": [42, 58],
      "classification": "email",
      "entity": "john@example.com"
    }
  ],
  "masked_email": "Hello, my name is [full_name] and my email is [email]. I need help with billing.",
  "category_of_the_email": "Request"
}
```

### GET /health

Health check endpoint to verify API status.

### GET /

Returns API information and available endpoints.

## Usage Example

```python
import requests

url = "https://HarshithGR5-email-classifier-api.hf.space/classify"
data = {
    "input_email_body": "Hi, I'm having technical issues with login. My phone is 555-123-4567."
}

response = requests.post(url, json=data)
print(response.json())
```

## Categories

- **Incident**: System outages, urgent issues
- **Request**: General requests, billing inquiries
- **Change**: Account updates, modifications
- **Problem**: Technical issues, bugs

## Technology Stack

- Flask API framework
- scikit-learn for machine learning
- spaCy for Named Entity Recognition
- Regular expressions for PII detection
- Docker for deployment
