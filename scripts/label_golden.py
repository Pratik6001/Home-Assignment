import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def main():
    print("Loading unlabelled golden set...")
    df = pd.read_csv('data/golden_eval_unlabelled.csv')
    
    # Cast to object so we can assign strings
    df['intent'] = df['intent'].astype(object)
    df['ideal_escalate'] = df['ideal_escalate'].astype(object)
    
    # Use gemini-3.6-flash which is very fast
    model = genai.GenerativeModel('gemini-3.6-flash')
    
    system_prompt = """
    You are an expert customer support annotator for SpotifyCares.
    Your task is to classify an array of customer tweets into intents:
    - Playback Issue: Issues playing music, buffering, skipping.
    - Login Issue: Password resets, hacked accounts, trouble signing in.
    - Premium Issue: Payment, subscription cancellation, family plan setup.
    - Feature Request: Asking for new features or songs.
    - Unknown/Other: General complaints, praises, or unclear issues.
    
    You also need to determine 'ideal_escalate'. Escalate (true) IF:
    - payment/billing issue.
    - account security/login block.
    - basic troubleshooting failed.
    Otherwise, escalate (false).
    
    IMPORTANT: You must reply ONLY with a raw valid JSON array of objects.
    Each object must have "id", "intent" (string), and "ideal_escalate" (boolean).
    Do NOT wrap it in markdown. Just the JSON array.
    """

    print("Beginning batch labelling process...")
    batch_size = 20
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        
        # Check if already labelled
        if pd.notna(batch.iloc[0]['intent']) and batch.iloc[0]['intent'] != '':
            continue
            
        messages = []
        for _, row in batch.iterrows():
            messages.append({"id": row['id'], "msg": row['customer_message']})
            
        prompt = f"{system_prompt}\n\nCustomer Messages:\n{json.dumps(messages, indent=2)}"
        
        success = False
        while not success:
            try:
                response = model.generate_content(prompt)
                text = response.text.strip()
                if text.startswith('```json'): text = text[7:]
                if text.startswith('```'): text = text[3:]
                if text.endswith('```'): text = text[:-3]
                
                results = json.loads(text.strip())
                
                for res in results:
                    idx = df[df['id'] == res['id']].index[0]
                    df.at[idx, 'intent'] = res['intent']
                    df.at[idx, 'ideal_escalate'] = res['ideal_escalate']
                    
                print(f"Batched labeled {i+len(batch)} / 200")
                df.to_csv('data/golden_eval_labelled.csv', index=False)
                success = True
                
                # Respect 5 RPM limit -> sleep 12s
                time.sleep(12)
                
            except Exception as e:
                print(f"Batch Error at {i}: {e}. Retrying in 15s...")
                time.sleep(15)
            
    print("Labelling complete. Saved to data/golden_eval_labelled.csv")

if __name__ == '__main__':
    main()
