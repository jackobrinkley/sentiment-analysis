from transformers import pipeline
import sys
import os

# Load sentiment model once (expensive operation)
sentiment_model = pipeline("sentiment-analysis")

def sentiment_to_stars(review: str) -> int:
    result = sentiment_model(review)[0]
    
    label = result['label']
    confidence = result['score']
    
    # Heuristic mapping → 1–5 stars
    if label == "NEGATIVE":
        if confidence > 0.9:
            return 1
        elif confidence > 0.7:
            return 2
        else:
            return 3
    else:  # POSITIVE
        if confidence > 0.9:
            return 5
        elif confidence > 0.7:
            return 4
        else:
            return 3

def read_review_from_file(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read().strip()
    
    if not content:
        raise ValueError("The file is empty.")
    
    return content

if __name__ == "__main__":
    # Default file name OR pass as argument
    filepath = sys.argv[1] if len(sys.argv) > 1 else "review.txt"
    
    try:
        review = read_review_from_file(filepath)
        stars = sentiment_to_stars(review)
        
        print(f"Review:\n{review}\n")
        print(f"Predicted Rating: {stars} stars")
    
    except Exception as e:
        print(f"Error: {e}")
