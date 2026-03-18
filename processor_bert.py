import joblib
import os
from sentence_transformers import SentenceTransformer

# Tier 2: BERT (Sentence Transformer) + Logistic Regression
# This handles complex logs that regex can't identify by understanding the meaning of text.

# Global variables to store models once loaded
_model_embedding = None
_model_classification = None

def get_bert_models():
    """
    Lazily loads the embedding and classification models.
    This prevents the server from timing out during startup on cloud platforms.
    """
    global _model_embedding, _model_classification
    
    if _model_embedding is None:
        print("Loading Sentence Transformer (BERT) model...")
        _model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
    
    if _model_classification is None:
        model_path = "models/log_classifier.joblib"
        if os.path.exists(model_path):
            print("Loading Logistic Regression classifier...")
            _model_classification = joblib.load(model_path)
        else:
            print(f"Warning: Trained model not found at {model_path}.")
            _model_classification = "MISSING" # Mark as missing so we don't try again
            
    return _model_embedding, _model_classification

def classify_with_bert(log_message):
    """
    Classify a log message by understanding its semantic meaning.
    It returns "Unclassified" if the AI isn't confident enough.
    """
    embedding_model, classifier_model = get_bert_models()
    
    if classifier_model is None or classifier_model == "MISSING":
        return "Unclassified (Model Missing)"

    # Step 1: Convert the log message into a mathematical representation (embedding)
    embeddings = embedding_model.encode([log_message])
    
    # Step 2: Get the confidence probabilities for each category
    probabilities = classifier_model.predict_proba(embeddings)[0]
    
    # Step 3: Check if the AI's highest confidence is above our threshold (50%)
    if max(probabilities) < 0.5:
        return "Unclassified"
    
    # Step 4: Pick the category with the highest confidence
    predicted_label = classifier_model.predict(embeddings)[0]
    
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
