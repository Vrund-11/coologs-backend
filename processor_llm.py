import os
import re
from dotenv import load_dotenv

load_dotenv()

# Global variable to store model once loaded
_gemini_model = None

def get_gemini_model():
    """
    Lazily loads the Gemini model only when needed.
    """
    global _gemini_model
    
    if _gemini_model is None:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            _gemini_model = genai.GenerativeModel("models/gemini-flash-latest")
        else:
            print("Warning: GEMINI_API_KEY not found in environment.")
            return None
            
    return _gemini_model

def classify_with_llm(log_msg):
    """
    Tier 3: Use Gemini AI to classify rare or complex logs.
    """
    model = get_gemini_model()
    if not model:
        return "Unclassified"

    prompt = f'''
    You are a log classification expert. 
    Classify the following log message into the MOST LIKELY category.
    
    Possible categories: 
    Workflow Error, Deprecation Warning, HTTP Status, Security Alert, System Notification, Critical Error, User Action, Resource Usage.
    
    Return result inside <category> </category> tags.
    
    Log message: {log_msg}
    '''

    try:
        response = model.generate_content(prompt)
        match = re.search(r'<category>(.*?)<\/category>', response.text, flags=re.DOTALL)
        if match:
            return match.group(1).strip()
        return "Unclassified"
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Unclassified"
