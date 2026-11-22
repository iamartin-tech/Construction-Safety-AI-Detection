# Construction Safety AI Project
# Part 3 - Gemini API Setup Test

import os
import time
from google import genai
from google.genai import errors

print("=" * 50)
print("GEMINI API SETUP TEST")
print("=" * 50)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found."
    )
    
print("API key envrionment variable found.")
print("API key value is hidden for security.")

# Create the Gemini client using the environment variable
client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-3.8-flash"
MAX_ATTEMPTS = 3

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        print(f"\nConnection attempt {attempt}/{MAX_ATTEMPTS}")
        print("Model:", MODEL_NAME)
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents="Reply with exactly: Gemini connection successful."
        )
        
        print("\nGemini response:")
        print(response.text)
        
        print("\nGEMINI API SETUP TEST COMPLETE")
        break
    
    except errors.APIError as error:
        print("\nAPI error:", error.code)
        print("Message:", error.message)
        
        if error.code in (429, 500, 502, 503, 504):
            if attempt < MAX_ATTEMPTS:
                print("Waiting 15 seconds before retrying...")
                time.sleep(15)
            else:
                print("All connection attempts failed.")
                print("Try again later or check API quotas.")
            
        else:
            print("This error is not automatically retryable.")
            break

client.close()

