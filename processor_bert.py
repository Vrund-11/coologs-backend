import os
import joblib

# Tier 2: BERT (Sentence Transformer) + Logistic Regression
# This handles complex logs that regex can't identify.

_model_embedding = None
_model_classification = None

def get_bert_models():
    """
    Lazily loads the heavy libraries and models only when needed.
    This prevents the server from timing out or crashing during startup on Render.
    """
    global _model_embedding, _model_classification
    
    if _model_embedding is None:
        # Move heavy import inside the function
        print("Initializing Sentence Transformer (BERT)...")
        from sentence_transformers import SentenceTransformer
        _model_embedding = SentenceTransformer('all-MiniLM-L6-v2')
    
    if _model_classification is None:
        model_path = "models/log_classifier.joblib"
        if os.path.exists(model_path):
            _model_classification = joblib.load(model_path)
        else:
            print(f"Warning: Trained model not found at {model_path}.")
            _model_classification = "MISSING"
            
    return _model_embedding, _model_classification

def classify_with_bert(log_message):
    embedding_model, classifier_model = get_bert_models()
    
    if classifier_model is None or classifier_model == "MISSING":
        return "Unclassified"

    embeddings = embedding_model.encode([log_message])
    probabilities = classifier_model.predict_proba(embeddings)[0]
    
    if max(probabilities) < 0.5:
        return "Unclassified"
    
    return classifier_model.predict(embeddings)[0]
