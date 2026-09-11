import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
import google.generativeai as genai
from scripts.agent import SupportAgent

def main():
    print("Loading environment variables...")
    load_dotenv()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Init agent with only 50 corpus items for instant speed
    print("Initializing Agent...")
    agent = SupportAgent('data/spotify_threads.csv', max_corpus_size=50)
    
    print("\n" + "="*50)
    print("🤖 SpotifyCares AI Agent - Interactive Test")
    print("Type 'quit' to exit.")
    print("="*50 + "\n")
    
    while True:
        user_input = input("Customer: ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
            
        print("Agent is typing...\n")
        try:
            response = agent.handle_query(user_input)
            print(f"Intent: {response['intent']}")
            print(f"Escalate: {response['escalate']}")
            if response['escalate']:
                print(f"Reason: {response['escalation_reason']}")
            print(f"\nReply: {response['reply']}")
        except Exception as e:
            print(f"API Error: {e}")
            print("Note: You may have hit your Free Tier daily quota for Gemini API.")
            
        print("\n" + "-"*50 + "\n")

if __name__ == '__main__':
    main()
