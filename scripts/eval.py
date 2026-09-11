import os
import json
import time
import random
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score
from agent import SupportAgent

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-3.6-flash')

class LLMJudgeScore(BaseModel):
    groundedness: int = Field(description="Score 1-5 on how well the reply solves the issue using historical context.")
    tone: int = Field(description="Score 1-5 on how well it matches the polite, helpful brand tone of SpotifyCares.")
    accuracy: int = Field(description="Score 1-5 on whether the reply avoids hallucinating fake policies.")
    feedback: str = Field(description="One sentence explaining the scores.")

def evaluate_reply(customer_msg: str, actual_reply: str, generated_reply: str) -> dict:
    prompt = f"""
    You are an expert evaluator grading an AI customer support agent for SpotifyCares.
    
    Customer Message: "{customer_msg}"
    Historical Ideal Reply: "{actual_reply}"
    AI Generated Reply: "{generated_reply}"
    
    Score the AI Generated Reply on a scale of 1-5 for:
    1. Groundedness (does it solve the problem using the historical context/ideal reply?)
    2. Tone (is it empathetic and matching a brand voice?)
    3. Accuracy (does it avoid hallucinations?)
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=LLMJudgeScore,
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Eval Error: {e}")
        return {"groundedness": 3, "tone": 3, "accuracy": 3, "feedback": "Error"}

def run_evaluation(num_samples=20):
    print(f"Loading golden set (Evaluating {num_samples} samples for speed)...")
    golden_df = pd.read_csv('data/golden_eval_labelled.csv').head(num_samples)
    
    # Init Agent
    agent = SupportAgent('data/spotify_threads.csv', max_corpus_size=50)
    
    # Get majority class for trivial baseline
    majority_intent = golden_df['intent'].mode()[0]
    
    results = {
        'trivial': {'intents': [], 'escalates': [], 'scores': []},
        'simple': {'intents': [], 'escalates': [], 'scores': []},
        'agent': {'intents': [], 'escalates': [], 'scores': []},
        'ground_truth': {'intents': golden_df['intent'].tolist(), 'escalates': golden_df['ideal_escalate'].tolist()}
    }

    print("Running evaluation (this may take a few minutes)...")
    for index, row in golden_df.iterrows():
        print(f"\nProcessing {index+1}/{num_samples}...")
        customer_msg = row['customer_message']
        actual_reply = row['actual_brand_reply']
        
        # --- 1. Trivial Baseline ---
        random_reply = agent.corpus_df.sample(1)['text_brand'].values[0]
        results['trivial']['intents'].append(majority_intent)
        results['trivial']['escalates'].append(True) # always escalate
        trivial_score = evaluate_reply(customer_msg, actual_reply, random_reply)
        results['trivial']['scores'].append(trivial_score)
        
        # --- 2. Simple Baseline (Zero-shot LLM without RAG) ---
        simple_prompt = f"""
        Classify intent and draft a reply for this SpotifyCares customer message:
        "{customer_msg}"
        Respond in JSON with keys: "reply", "escalate" (bool), "intent"
        """
        try:
            # We reuse the agent's schema for simplicity
            from agent import AgentResponse
            simple_resp = model.generate_content(
                simple_prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=AgentResponse,
                )
            )
            simple_data = json.loads(simple_resp.text)
        except:
            simple_data = {"reply": "Sorry, I can't help.", "escalate": True, "intent": "Unknown/Other"}
            
        results['simple']['intents'].append(simple_data['intent'])
        results['simple']['escalates'].append(simple_data['escalate'])
        simple_score = evaluate_reply(customer_msg, actual_reply, simple_data['reply'])
        results['simple']['scores'].append(simple_score)
        
        # --- 3. RAG Agent ---
        try:
            agent_data = agent.handle_query(customer_msg)
        except:
            agent_data = {"reply": "Sorry, I can't help.", "escalate": True, "intent": "Unknown/Other"}
            
        results['agent']['intents'].append(agent_data['intent'])
        results['agent']['escalates'].append(agent_data['escalate'])
        agent_score = evaluate_reply(customer_msg, actual_reply, agent_data['reply'])
        results['agent']['scores'].append(agent_score)
        
        time.sleep(12) # Rate limit protection

    print("\n\n--- EVALUATION RESULTS ---")
    
    # Calculate metrics
    def calc_metrics(approach):
        acc = accuracy_score(results['ground_truth']['intents'], results[approach]['intents'])
        f1 = f1_score(results['ground_truth']['escalates'], results[approach]['escalates'], zero_division=0)
        
        avg_groundedness = sum(s['groundedness'] for s in results[approach]['scores']) / num_samples
        avg_tone = sum(s['tone'] for s in results[approach]['scores']) / num_samples
        avg_accuracy = sum(s['accuracy'] for s in results[approach]['scores']) / num_samples
        
        return acc, f1, avg_groundedness, avg_tone, avg_accuracy

    for name in ['trivial', 'simple', 'agent']:
        acc, f1, g, t, a = calc_metrics(name)
        print(f"\n{name.upper()}:")
        print(f"Intent Accuracy: {acc:.2f}")
        print(f"Escalate F1: {f1:.2f}")
        print(f"Reply Judge Scores -> Grounded: {g:.2f}/5, Tone: {t:.2f}/5, Accuracy: {a:.2f}/5")

if __name__ == '__main__':
    run_evaluation(num_samples=15)
