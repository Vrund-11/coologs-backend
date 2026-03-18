import joblib
import os
from sentence_transformers import SentenceTransformer

# Tier 2: BERT (Sentence Transformer) + Logistic Regression
# This handles complex logs that regex can't identify by understanding the meaning of text.

# Load the Sentence Transformer model (converts text to numbers/embeddings)
# Using a lightweight model for speed
model_embedding = SentenceTransformer('all-MiniLM-L6-v2')

# Load the trained Logistic Regression model
# We check if the model file exists first to avoid errors
model_path = "models/log_classifier.joblib"
if os.path.exists(model_path):
    model_classification = joblib.load(model_path)
else:
    model_classification = None
    print(f"Warning: Trained model not found at {model_path}. Please run train_model.py first.")

def classify_with_bert(log_message):
    """
    Classify a log message by understanding its semantic meaning.
    It returns "Unclassified" if the AI isn't confident enough.
    """
    if model_classification is None:
        return "Unclassified (Model Missing)"

    # Step 1: Convert the log message into a mathematical representation (embedding)
    embeddings = model_embedding.encode([log_message])
    
    # Step 2: Get the confidence probabilities for each category
    probabilities = model_classification.predict_proba(embeddings)[0]
    
    # Step 3: Check if the AI's highest confidence is above our threshold (50%)
    # This prevents the AI from just "guessing" a random category
    if max(probabilities) < 0.5:
        return "Unclassified"
    
    # Step 4: Pick the category with the highest confidence
    predicted_label = model_classification.predict(embeddings)[0]
    
    return predicted_label

if __name__ == "__main__":
    # Test cases to check the BERT classification
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!", # Should be unclassified
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(f"Log: {log} \n-> Prediction: {label}\n")
