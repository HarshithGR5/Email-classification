#!/usr/bin/env python3
"""
Email Classification Model with loading capabilities
"""

import pandas as pd
import numpy as np
import os
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class EmailClassifierTrainer:
    def __init__(self):
        self.pipeline = None
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # If NLTK stopwords not available, use basic list
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                             'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text) or text == "":
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove emails, URLs, and special characters
        text = re.sub(r'\S+@\S+', ' ', text)  # Remove emails
        text = re.sub(r'http\S+|www\S+', ' ', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters and spaces
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        
        # Tokenize and remove stopwords
        try:
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) 
                     for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        except:
            # Fallback if NLTK fails
            tokens = text.split()
            tokens = [token for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)

    def load_model(self, model_path='models/email_classifier.pkl'):
        """Load a pre-trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.pipeline = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")
        return True
    
    def predict(self, email_text):
        """Predict the category of an email"""
        if self.pipeline is None:
            raise ValueError("Model not loaded. Please load a model first using load_model()")
        
        # Preprocess the email
        processed_email = self.preprocess_text(email_text)
        
        if not processed_email:
            # If preprocessing results in empty text, return default
            return "Request", 0.5
        
        # Make prediction
        prediction = self.pipeline.predict([processed_email])[0]
        
        # Get prediction probabilities for confidence
        probabilities = self.pipeline.predict_proba([processed_email])
        confidence = np.max(probabilities)
        
        return prediction, confidence
    
    def load_dataset(self, csv_path):
        """Load and validate dataset"""
        print(f"Loading dataset from: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found: {csv_path}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        
        # Check required columns
        if 'email' not in df.columns or 'type' not in df.columns:
            raise ValueError("Dataset must have 'email' and 'type' columns")
        
        # Remove any rows with missing data
        df = df.dropna(subset=['email', 'type'])
        print(f"After removing missing data: {df.shape}")
        
        # Show class distribution
        print("\nClass distribution:")
        class_counts = df['type'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
        
        return df
    
    def train_model(self, df):
        """Train the classification model"""
        print("Starting model training...")
        
        # Preprocess emails
        print("Preprocessing emails...")
        df['processed_email'] = df['email'].apply(self.preprocess_text)
        
        # Remove empty processed emails
        df = df[df['processed_email'].str.len() > 0]
        print(f"After preprocessing: {df.shape}")
        
        # Prepare data
        X = df['processed_email']
        y = df['type']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        
        # Train model
        print("Training Random Forest classifier...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        classes = sorted(y.unique())
        print("Predicted ->", " ".join(f"{cls:>10}" for cls in classes))
        for i, actual_class in enumerate(classes):
            print(f"{actual_class:>9} |", " ".join(f"{cm[i][j]:>10}" for j in range(len(classes))))
        
        return accuracy
    
    def save_model(self, model_dir='models'):
        """Save the trained model"""
        if self.pipeline is None:
            raise ValueError("Model not trained yet!")
        
        # Create models directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the complete pipeline
        model_path = os.path.join(model_dir, 'email_classifier.pkl')
        joblib.dump(self.pipeline, model_path)
        print(f"Model saved to: {model_path}")
        
        # Also save individual components for flexibility
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        classifier_path = os.path.join(model_dir, 'random_forest_classifier.pkl')
        
        joblib.dump(self.pipeline.named_steps['tfidf'], vectorizer_path)
        joblib.dump(self.pipeline.named_steps['classifier'], classifier_path)
        
        print(f"Vectorizer saved to: {vectorizer_path}")
        print(f"Classifier saved to: {classifier_path}")
        
        return model_path

def main():
    print("="*60)
    print("EMAIL CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # Get dataset path
    dataset_path = input("Enter path to your CSV dataset file: ").strip()
    
    if not dataset_path:
        # Try common locations
        common_paths = [
            'data/support_emails.csv',
            'support_emails.csv',
            r'C:\Projects\akaike project\combined_emails_with_natural_pii.csv',
            'data/dataset.csv'
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                dataset_path = path
                print(f"Found dataset at: {dataset_path}")
                break
        
        if not dataset_path:
            print("No dataset found. Please provide the path.")
            return
    
    try:
        # Initialize trainer
        trainer = EmailClassifierTrainer()
        
        # Load dataset
        df = trainer.load_dataset(dataset_path)
        
        # Train model
        accuracy = trainer.train_model(df)
        
        # Save model
        model_path = trainer.save_model()
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final Accuracy: {accuracy:.4f}")
        print(f"Model saved to: {model_path}")
        print("Ready to use in your Flask API!")
        print("="*60)
        
        # Test the saved model
        print("Testing saved model...")
        loaded_pipeline = joblib.load(model_path)
        
        # Test with a sample email
        test_email = "Hello, I'm having trouble with my billing account and need help"
        processed_test = trainer.preprocess_text(test_email)
        prediction = loaded_pipeline.predict([processed_test])[0]
        confidence = loaded_pipeline.predict_proba([processed_test]).max()
        
        print(f"Test email: {test_email}")
        print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
        print("Model loading and prediction test successful!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        print("Please check your dataset format and try again.")

if __name__ == "__main__":
    main()