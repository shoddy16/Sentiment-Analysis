"""
Prediction Script for Sentiment Analysis
========================================

This script provides an easy-to-use interface for making sentiment predictions
using trained models. It supports both single text predictions and batch processing.

Features:
---------
- Simple command-line interface
- Batch prediction from CSV files
- Confidence scores for predictions
- Output formatting options
- Integration with trained models

Author: AI Assistant
Python Version: 3.12+
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from typing import List, Dict, Union, Optional
import sys
import os

# Import our custom modules
from .sentiment_analyzer import SentimentAnalyzer
from .text_preprocessor import TextPreprocessor


class SentimentPredictor:
    """
    Easy-to-use prediction class for sentiment analysis.
    
    This class provides simple methods for making predictions with trained models,
    making it perfect for deployment and practical use.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the SentimentPredictor.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to a trained model file. If None, you'll need to train a model first.
            
        Example:
        --------
        >>> # Load a pre-trained model
        >>> predictor = SentimentPredictor('path/to/model.pkl')
        >>> 
        >>> # Or create a new one and train it
        >>> predictor = SentimentPredictor()
        >>> # ... train model with data ...
        """
        self.model = None
        self.vectorizer = None
        self.is_loaded = False
        self.preprocessor = TextPreprocessor()
        
        if model_path:
            self.load_model(model_path)
    
    @staticmethod
    def _readable_label(label, classes=None):
        """Convert common binary and three-class sentiment labels to names."""
        if isinstance(label, str):
            value = label.strip().lower()
            aliases = {
                "pos": "positive", "positive": "positive",
                "neg": "negative", "negative": "negative",
                "neu": "neutral", "neutral": "neutral",
            }
            return aliases.get(value, value)

        if isinstance(label, (int, float, np.integer, np.floating)):
            class_values = set(classes) if classes is not None else set()
            if class_values == {0, 1}:
                return "positive" if label == 1 else "negative"
            if class_values == {0, 1, 2}:
                return {0: "negative", 1: "neutral", 2: "positive"}.get(label, str(label))
            if class_values == {0, 2, 4}:
                return {0: "negative", 2: "neutral", 4: "positive"}.get(label, str(label))
            if class_values == {-1, 0, 1}:
                return {-1: "negative", 0: "neutral", 1: "positive"}.get(label, str(label))
        return str(label)

    @classmethod
    def _readable_confidences(cls, classes, probabilities):
        """Return probabilities keyed by readable sentiment labels."""
        return {
            cls._readable_label(label, classes): float(probability)
            for label, probability in zip(classes, probabilities)
        }

    def load_model(self, model_path: str):
        """
        Load a trained model from file.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained model file (.pkl)
            
        Example:
        --------
        >>> predictor = SentimentPredictor()
        >>> predictor.load_model('my_trained_model.pkl')
        >>> result = predictor.predict_text("This is great!")
        """
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            self.is_loaded = True
            print(f"✓ Model loaded successfully from {model_path}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def predict_text(self, text: str, return_confidence: bool = False) -> Union[str, Dict[str, float]]:
        """
        Predict sentiment for a single text.
        
        This is the simplest way to make a prediction. Just pass any text and
        get the sentiment result.
        
        Parameters:
        -----------
        text : str
            The text to analyze
        return_confidence : bool
            Whether to return confidence scores along with prediction
            
        Returns:
        --------
        str or Dict[str, float]
            Sentiment prediction (and confidence if requested)
            
        Example:
        --------
        >>> predictor = SentimentPredictor('model.pkl')
        >>> 
        >>> # Simple prediction
        >>> result = predictor.predict_text("This movie is amazing!")
        >>> print(f"Sentiment: {result}")  # Output: Sentiment: positive
        >>> 
        >>> # Prediction with confidence
        >>> result = predictor.predict_text("This movie is amazing!", return_confidence=True)
        >>> print(f"Prediction: {result}")  # Output: Prediction: {'positive': 0.92, 'negative': 0.08}
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Please load a trained model first.")
        
        # Clean the input text
        cleaned_text = self.preprocessor.basic_clean(text)
        
        # Make prediction
        prediction = self.model.predict([cleaned_text])[0]
        
        # Preserve the model's class labels, including neutral when trained.
        sentiment = self._readable_label(prediction, getattr(self.model, "classes_", None))
        
        if return_confidence and hasattr(self.model, 'predict_proba'):
            # Get confidence scores
            probabilities = self.model.predict_proba([cleaned_text])[0]
            classes = self.model.classes_
            
            confidence_dict = self._readable_confidences(classes, probabilities)
            
            return confidence_dict
        
        return sentiment
    
    def predict_batch(self, texts: List[str], return_confidence: bool = False) -> List[Union[str, Dict[str, float]]]:
        """
        Predict sentiment for multiple texts at once.
        
        This method is more efficient than calling predict_text multiple times
        because it processes all texts together.
        
        Parameters:
        -----------
        texts : List[str]
            List of texts to analyze
        return_confidence : bool
            Whether to return confidence scores
            
        Returns:
        --------
        List[str] or List[Dict[str, float]]
            List of predictions (with confidence if requested)
            
        Example:
        --------
        >>> texts = ["Great product!", "Terrible service", "It's okay"]
        >>> results = predictor.predict_batch(texts)
        >>> for text, sentiment in zip(texts, results):
        ...     print(f"'{text}' -> {sentiment}")
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Please load a trained model first.")
        
        # Clean all texts
        cleaned_texts = [self.preprocessor.basic_clean(text) for text in texts]
        
        # Make predictions
        predictions = self.model.predict(cleaned_texts)
        
        # Convert labels using the model's complete class set.
        classes = getattr(self.model, "classes_", None)
        results = [
            self._readable_label(prediction, classes)
            for prediction in predictions
        ]
        
        if return_confidence and hasattr(self.model, 'predict_proba'):
            # Get confidence scores
            probabilities = self.model.predict_proba(cleaned_texts)
            classes = self.model.classes_
            
            confidence_results = [
                self._readable_confidences(classes, prob)
                for prob in probabilities
            ]
            
            return confidence_results
        
        return results
    
    def predict_from_csv(self, csv_path: str, text_column: str,
                        output_path: Optional[str] = None,
                        return_confidence: bool = False) -> pd.DataFrame:
        """
        Predict sentiment for texts in a CSV file.
        
        This method is perfect for batch processing large datasets.
        It reads a CSV file, makes predictions, and saves the results.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing texts to analyze
        text_column : str
            Name of the column containing the texts
        output_path : str, optional
            Path to save the results (if None, just returns DataFrame)
        return_confidence : bool
            Whether to include confidence scores in output
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with original data and sentiment predictions
            
        Example:
        --------
        >>> # Predict sentiments for reviews in a CSV file
        >>> results = predictor.predict_from_csv('reviews.csv', 'review_text')
        >>> print(results[['review_text', 'sentiment']].head())
        >>> 
        >>> # Save results to new CSV
        >>> predictor.predict_from_csv('reviews.csv', 'review_text', 
        ...                           'reviews_with_sentiment.csv')
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Please load a trained model first.")
        
        # Load the CSV file
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Check if the text column exists
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. "
                           f"Available columns: {list(df.columns)}")
        
        # Get texts to predict
        texts = df[text_column].tolist()
        
        print(f"Making predictions for {len(texts)} texts...")
        
        # Make predictions
        if return_confidence:
            predictions = self.predict_batch(texts, return_confidence=True)
            
            # Preserve all model classes, including neutral, in CSV output.
            if predictions and all(isinstance(pred, dict) for pred in predictions):
                labels = sorted({label for pred in predictions for label in pred})
                df['sentiment'] = [
                    max(pred, key=pred.get) if pred else 'unknown'
                    for pred in predictions
                ]

                for label in labels:
                    df[f'confidence_{label}'] = [
                        pred.get(label, 0.0) for pred in predictions
                    ]

                df['confidence'] = [
                    max(pred.values(), default=0.0) for pred in predictions
                ]
            else:
                # Models without predict_proba return labels, not confidence dictionaries.
                df['sentiment'] = predictions
        else:
            predictions = self.predict_batch(texts, return_confidence=False)
            df['sentiment'] = predictions
        
        # Save results if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"✓ Results saved to {output_path}")
        
        print(f"✓ Predictions completed for {len(texts)} texts")
        
        return df
    
    def interactive_predict(self):
        """
        Start an interactive prediction session.
        
        This method provides a simple command-line interface where users can
        type text and get immediate sentiment predictions.
        
        Example:
        --------
        >>> predictor = SentimentPredictor('model.pkl')
        >>> predictor.interactive_predict()
        
        Interactive Sentiment Predictor
        Type 'quit' to exit
        
        Enter text: This movie is amazing!
        Sentiment: positive (confidence: 94.2%)
        
        Enter text: I didn't like the plot
        Sentiment: negative (confidence: 87.5%)
        
        Enter text: quit
        Goodbye!
        """
        if not self.is_loaded:
            raise ValueError("No model loaded. Please load a trained model first.")
        
        print("\n" + "="*50)
        print("Interactive Sentiment Predictor")
        print("="*50)
        print("Type 'quit' to exit\n")
        
        while True:
            try:
                # Get input from user
                user_input = input("Enter text: ").strip()
                
                # Check if user wants to quit
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Make prediction with confidence
                confidence = self.predict_text(user_input, return_confidence=True)
                
                if isinstance(confidence, dict):
                    positive_conf = confidence.get('positive', 0)
                    negative_conf = confidence.get('negative', 0)
                    
                    if positive_conf > negative_conf:
                        sentiment = 'positive'
                        conf_percent = positive_conf * 100
                    else:
                        sentiment = 'negative'
                        conf_percent = negative_conf * 100
                    
                    print(f"Sentiment: {sentiment} (confidence: {conf_percent:.1f}%)")
                else:
                    print(f"Sentiment: {confidence}")
                
                print()
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
                print()
    
    def get_prediction_summary(self, csv_path: str, text_column: str) -> Dict[str, any]:
        """
        Get a summary of predictions for a dataset.
        
        This method provides useful statistics about sentiment distribution
        in a dataset.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file with texts to analyze
        text_column : str
            Name of the column containing texts
            
        Returns:
        --------
        Dict[str, any]
            Summary statistics about predictions
            
        Example:
        --------
        >>> summary = predictor.get_prediction_summary('reviews.csv', 'review_text')
        >>> print(f"Positive reviews: {summary['positive_count']} ({summary['positive_percentage']:.1f}%)")
        """
        # Get predictions
        results_df = self.predict_from_csv(csv_path, text_column)
        
        # Calculate statistics
        total_predictions = len(results_df)
        sentiment_counts = results_df['sentiment'].value_counts()
        
        summary = {
            'total_predictions': total_predictions,
            'sentiment_counts': sentiment_counts.to_dict(),
            'positive_count': sentiment_counts.get('positive', 0),
            'negative_count': sentiment_counts.get('negative', 0),
            'positive_percentage': (sentiment_counts.get('positive', 0) / total_predictions) * 100,
            'negative_percentage': (sentiment_counts.get('negative', 0) / total_predictions) * 100
        }
        
        return summary
    
    def print_summary(self, csv_path: str, text_column: str):
        """
        Print a formatted summary of predictions.
        
        Parameters:
        -----------
        csv_path : str
            Path to CSV file with texts to analyze
        text_column : str
            Name of the column containing texts
        """
        summary = self.get_prediction_summary(csv_path, text_column)
        
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        print(f"Total texts analyzed: {summary['total_predictions']:,}")
        print()
        print("Sentiment Distribution:")
        print(f"• Positive: {summary['positive_count']:,} ({summary['positive_percentage']:.1f}%)")
        print(f"• Negative: {summary['negative_count']:,} ({summary['negative_percentage']:.1f}%)")
        print("="*50)


def main():
    """
    Example usage of the SentimentPredictor class.
    
    This function demonstrates different ways to use the prediction functionality.
    """
    print("Sentiment Prediction Script - Example Usage")
    print("="*60)
    
    print("\n1. Basic Usage:")
    print("-" * 30)
    print("# Load a trained model and make predictions")
    print("predictor = SentimentPredictor('trained_model.pkl')")
    print("result = predictor.predict_text('This is amazing!')")
    print("print(f'Sentiment: {result}')")
    
    print("\n2. Batch Predictions:")
    print("-" * 30)
    print("# Predict multiple texts at once")
    print("texts = ['Great product!', 'Terrible service', 'It is okay']")
    print("results = predictor.predict_batch(texts)")
    print("for text, sentiment in zip(texts, results):")
    print("    print(f'{text} -> {sentiment}')")
    
    print("\n3. CSV Processing:")
    print("-" * 30)
    print("# Process entire CSV file")
    print("results_df = predictor.predict_from_csv(")
    print("    'reviews.csv', 'review_text', 'results.csv')")
    
    print("\n4. Interactive Mode:")
    print("-" * 30)
    print("# Start interactive prediction session")
    print("predictor.interactive_predict()")
    
    print("\n5. Confidence Scores:")
    print("-" * 30)
    print("# Get prediction confidence")
    print("confidence = predictor.predict_text(")
    print("    'This is great!', return_confidence=True)")
    print("print(f'Confidence: {confidence}')")
    
    print("\n" + "="*60)
    print("SentimentPredictor is ready to use!")
    print("="*60)


if __name__ == "__main__":
    main()