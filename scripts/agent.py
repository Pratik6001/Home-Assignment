import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel, Field

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define structured output for the Agent
class AgentResponse(BaseModel):
    reply: str = Field(description="The drafted response to the customer matching the brand's historical tone.")
    escalate: bool = Field(description="True if the issue requires a human agent (e.g., account security, billing disputes, complex unresolved bugs).")
    escalation_reason: str = Field(description="If escalate is true, explain why. If false, output an empty string.")
    intent: str = Field(description="The classified intent of the customer's query.")

class SupportAgent:
    def __init__(self, corpus_path: str, max_corpus_size: int = 500):
        print(f"Initializing Support Agent with {max_corpus_size} historical interactions...")
        self.corpus_df = pd.read_csv(corpus_path).head(max_corpus_size)
        self.embedding_model = "models/gemini-embedding-2"
        self.gen_model = genai.GenerativeModel('gemini-3.6-flash')
        
        # Build the RAG Corpus
        self._build_vector_store()

    def _build_vector_store(self):
        print("Building vector store (Embedding historical queries)...")
        # We embed the customer's message to match similar incoming queries
        queries = self.corpus_df['text_customer'].tolist()
        
        # Gemini embedding API accepts batches. We'll batch in size 100.
        embeddings = []
        batch_size = 100
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            success = False
            while not success:
                try:
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=batch,
                        task_type="retrieval_document",
                    )
                    embeddings.extend(result['embedding'])
                    success = True
                except Exception as e:
                    print(f"Embedding rate limit hit, waiting 30s... ({e})")
                    import time
                    time.sleep(30)
            
        self.corpus_embeddings = np.array(embeddings)
        print("Vector store ready!")

    def retrieve(self, query: str, top_k: int = 3):
        # Embed the query
        result = genai.embed_content(
            model=self.embedding_model,
            content=query,
            task_type="retrieval_query",
        )
        query_embedding = np.array(result['embedding'])
        
        # Calculate Cosine Similarity
        # Dot product works since embeddings are normalized, but let's do explicit cosine similarity just in case
        norms = np.linalg.norm(self.corpus_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = np.dot(self.corpus_embeddings, query_embedding) / (norms * query_norm)
        
        # Get top K indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Fetch results
        retrieved = []
        for idx in top_indices:
            row = self.corpus_df.iloc[idx]
            retrieved.append({
                'customer_message': row['text_customer'],
                'brand_reply': row['text_brand'],
                'similarity': similarities[idx]
            })
            
        return retrieved

    def handle_query(self, customer_query: str) -> dict:
        # 1. Retrieve similar past interactions
        context_items = self.retrieve(customer_query, top_k=3)
        
        context_str = ""
        for i, item in enumerate(context_items):
            context_str += f"--- Example {i+1} ---\n"
            context_str += f"Customer: {item['customer_message']}\n"
            context_str += f"Brand Reply: {item['brand_reply']}\n\n"
            
        # 2. Construct Prompt for the LLM
        prompt = f"""
You are an expert AI customer support agent for SpotifyCares.
Your task is to classify the intent of the incoming customer message, draft a helpful reply matching the brand's tone, and decide if this issue needs human escalation.

Here are some historically successful resolutions for similar queries to use as guidance for your tone and solutions:
{context_str}

Customer Query: "{customer_query}"

Instructions:
1. Classify the intent (Playback Issue, Login Issue, Premium Issue, Feature Request, or Unknown/Other).
2. Draft a 'reply' to the customer. It should be empathetic, concise, and helpful. Use the historical examples for inspiration on tone and typical troubleshooting steps (e.g. asking for DM, device details, or providing an FAQ link).
3. Set 'escalate' to true ONLY if it's a security issue, payment/billing dispute, or if standard troubleshooting has failed and human verification is needed. Otherwise, handle it automatically.
"""
        # 3. Call LLM with structured output
        response = self.gen_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=AgentResponse,
                temperature=0.3
            )
        )
        
        return json.loads(response.text)

if __name__ == '__main__':
    # Simple interactive test
    agent = SupportAgent('data/spotify_threads.csv', max_corpus_size=50)
    print("\n--- Support Agent Ready ---")
    test_query = "My spotify randomly pauses when I'm listening on my iphone, help!"
    print(f"Customer: {test_query}")
    response = agent.handle_query(test_query)
    print("\nAgent Response:")
    print(json.dumps(response, indent=2))
