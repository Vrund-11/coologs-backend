import os
import re
import google.generativeai as genai
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Configure the Gemini API with your key
# Get a free key at https://aistudio.google.com/
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("Warning: GEMINI_API_KEY not found in environment variables.")

# Use the stable fast model: gemini-flash-latest
model = genai.GenerativeModel("models/gemini-flash-latest")

def classify_with_llm(log_msg):
    """
    Tier 3: Use Gemini AI to classify rare or complex logs.
    This is used as a fallback when Regex and BERT fail.
    """
    # Define the prompt for the AI
    prompt = f'''
    You are a log classification expert. 
    Classify the following log message into the MOST LIKELY category.
    
    Possible categories: 
    Workflow Error, Deprecation Warning, HTTP Status, Security Alert, System Notification, Critical Error, User Action, Resource Usage.
    
    If it doesn't fit any of those, provide a NEW category name that fits.
    
    Return the result inside <category> </category> tags.
    
    Log message: {log_msg}
    '''

    try:
        # Ask Gemini to classify the log
        response = model.generate_content(prompt)
        content = response.text
        
        # Find the category inside the <category> tags
        match = re.search(r'<category>(.*?)<\/category>', content, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no tags are found, just return the raw text (cleaned up)
        return content.strip()
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Unclassified"

if __name__ == "__main__":
    # Test cases
    print("Testing with Workflow Error log:")
    print(classify_with_llm("Case escalation for ticket ID 7324 failed because the assigned support agent is no longer active."))
    
    print("\nTesting with Deprecation Warning log:")
    print(classify_with_llm("The 'ReportGenerator' module will be retired in version 4.0. Please migrate to the 'AdvancedAnalyticsSuite' by Dec 2025"))
