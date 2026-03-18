import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import re

def train():
    df = pd.read_csv("training/dataset/synthetic_logs.csv")
    
    def classify_with_regex(log_message):
        regex_patterns = {
            r"User User\d+ logged (in|out).": "User Action",
            r"Backup (started|ended) at .*": "System Notification",
            r"Backup completed successfully.": "System Notification",
            r"System updated to version .*": "System Notification",
            r"File .* uploaded successfully by user .*": "System Notification",
            r"Disk cleanup completed successfully.": "System Notification",
            r"System reboot initiated by user .*": "System Notification",
            r"Account with ID .* created by .*": "User Action"
        }
        for pattern, label in regex_patterns.items():
            if re.search(pattern, log_message):
                return label
        return None

    # Apply regex
    df['regex_label'] = df['log_message'].apply(classify_with_regex)
    
    # Filter logs that regex couldn't classify
    df_non_regex = df[df['regex_label'].isnull()].copy()
    
    # Separate LegacyCRM
    df_legacy = df_non_regex[df_non_regex['source'] == 'LegacyCRM'].copy()
    df_bert = df_non_regex[df_non_regex['source'] != 'LegacyCRM'].copy()
    
    # Encode with BERT
    print("Encoding logs...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X = model.encode(df_bert['log_message'].tolist())
    y = df_bert['target_label'].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    print("Accuracy:", clf.score(X_test, y_test))
    
    # Save the model
    joblib.dump(clf, "models/log_classifier.joblib")
    print("Model saved to models/log_classifier.joblib")

if __name__ == "__main__":
    train()
