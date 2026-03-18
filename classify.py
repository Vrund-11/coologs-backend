import pandas as pd
from processor_regex import classify_with_regex
from processor_bert import classify_with_bert
from processor_llm import classify_with_llm

# This is the Master Router (Mastermind)
# It decides which "Tier" should handle a log message.

def classify_log(source, log_msg):
    """
    Route a single log message through our 3-tier hybrid system.
    """
    # Step 1: Check the Source. 
    # If it's a legacy system with very rare logs, go straight to Gemini AI.
    if source == "LegacyCRM":
        return classify_with_llm(log_msg)
    
    # Step 2: Tier 1 - Use Regex (Fastest)
    label = classify_with_regex(log_msg)
    
    # Step 3: Tier 2 - If Regex fails, use BERT AI (Medium Speed)
    if not label or label == "Unknown":
        label = classify_with_bert(log_msg)
        
    # Step 4: Tier 3 - If BERT fails (unclassified), fallback to Gemini AI (Slow but Smart)
    if label == "Unclassified":
        label = classify_with_llm(log_msg)
        
    return label

def classify(logs):
    """
    Process a list of log messages.
    Each entry in the list should be a tuple of (source, log_message).
    """
    labels = []
    # Loop through each log and classify it using our routing logic
    for source, log_msg in logs:
        label = classify_log(source, log_msg)
        labels.append(label)
    return labels

def classify_csv(input_file):
    """
    Load a CSV file, classify every log in it, and save it to a new file.
    """
    # Load the data using Pandas
    df = pd.read_csv(input_file)

    # Perform classification for every row
    # We zip 'source' and 'log_message' columns together
    df["target_label"] = classify(list(zip(df["source"], df["log_message"])))

    # Save the modified file to 'output.csv'
    output_file = "output.csv"
    df.to_csv(output_file, index=False)

    return output_file

if __name__ == '__main__':
    # You can test the script by running it directly
    # Just make sure a 'test.csv' exists in the folder
    import os
    if os.path.exists("test.csv"):
        result = classify_csv("test.csv")
        print(f"Success! Logs classified and saved to {result}")
    else:
        print("Please create a 'test.csv' file with 'source' and 'log_message' columns.")
