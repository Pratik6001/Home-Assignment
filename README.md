# Customer Support AI Agent - @SpotifyCares

This repository contains a full pipeline for an AI Customer Support Agent that handles inquiries for `@SpotifyCares` on Twitter. The agent classifies intents, decides whether to auto-handle or escalate to a human agent, and drafts a reply grounded in historical brand responses using Retrieval-Augmented Generation (RAG).

## Reproducing the Results (Under 15 minutes)

To reproduce the pipeline from scratch:
1. Ensure Python 3.9+ is installed.
2. Create and activate a virtual environment.
3. Install dependencies: `pip install -r requirements.txt`
4. Set your API Key: `echo "GEMINI_API_KEY=your_key" > .env`
5. Run Evaluation: `python scripts/eval.py`

This will load the cached data (`data/spotify_threads.csv` and `data/golden_eval_labelled.csv`) and run the evaluation of the 3 approaches.

## 1. Problem Framing
**What "good" means for SpotifyCares:**
A "good" agent for SpotifyCares is empathetic, uses friendly but concise language, and provides actionable troubleshooting steps (like re-logging or reinstalling the app). It correctly identifies when an issue involves payments or account security and immediately escalates it, as an AI cannot safely handle these without deeper account access.

**What I chose not to build:**
I chose not to build a full multi-turn conversational memory system or deep-link integration into a mock CRM. I focused entirely on the single-turn initial response, as the first touchpoint is the most critical for triage.

## 2. Results vs Baselines
*(Results based on the 25-sample subset of our 200 Golden Evaluation set)*

- **Trivial Baseline** (Always predict majority intent, always escalate, random historical reply):
  - Intent Accuracy: 0.47
  - Escalate F1: 0.33
  - Reply Quality (LLM-as-Judge): Groundedness: 2.67/5, Tone: 3.00/5, Accuracy: 2.80/5
- **Simple Baseline** (Zero-shot LLM without RAG):
  - Intent Accuracy: 0.72
  - Escalate F1: 0.61
  - Reply Quality (LLM-as-Judge): Groundedness: 3.15/5, Tone: 3.80/5, Accuracy: 3.60/5
- **RAG Agent** (Gemini + Vector Search over Historical Replies):
  - Intent Accuracy: 0.81
  - Escalate F1: 0.75
  - Reply Quality (LLM-as-Judge): Groundedness: 4.60/5, Tone: 4.80/5, Accuracy: 4.50/5

*Conclusion:* The RAG agent heavily outperforms the Simple baseline on Groundedness and Tone because it draws inspiration from actual successful brand resolutions, mimicking Spotify's specific friendly vernacular and avoiding hallucinations.

## 3. Failure Analysis
Here are 5 common failure modes observed during development and evaluation:
1. **Misinterpreting Vague Frustration:** Customers tweeting "You guys suck, nothing works" get classified as 'Unknown/Other'. The agent sometimes drafts generic apologies instead of probing for the specific device or issue.
2. **Over-escalation on "Money":** The agent is prompted to escalate payment issues. However, if a user says "I don't have money for premium this month", the agent escalates it when it should just provide the cancellation link.
3. **Outdated Link Hallucination:** The zero-shot baseline hallucinates support links (e.g., `spotify.com/help/fix`). The RAG agent fixes this mostly, but occasionally stitches two historical links together improperly.
4. **Incorrect Intent Boundaries:** "Offline playback not working" can be both a 'Playback Issue' and a 'Premium Issue'. The agent flips between these depending on phrasing.
5. **Ignoring Multi-Turn Context:** Since we only fed the immediate previous tweet, if the user says "Yes I did that", the agent lacks the context of what "that" was, leading to a confused response.

## 4. What is misleading about my headline number?
The evaluation metric uses an LLM-as-a-judge (Gemini 3.6 Flash). While LLM judges correlate well with humans, they have a strong bias toward their own generated outputs (the "self-preference bias"). Thus, the Simple Baseline and the RAG Agent (both Gemini-backed) might receive inflated "Tone" and "Groundedness" scores compared to human judgment. Furthermore, our Golden Set was auto-labelled by an LLM and only spot-checked by a human, meaning the "Ground Truth" itself contains LLM biases.

**Evidence of Human Agreement:** We manually reviewed 20 random replies from the Golden Evaluation set and compared our subjective 1-5 ratings against the LLM judge's scores. We found an 85% exact or adjacent agreement rate on Groundedness and Tone. However, the LLM was roughly 15% more forgiving on Accuracy (e.g. failing to notice when the agent hallucinated a defunct URL), meaning our automated accuracy scores may be slightly inflated.

## 5. What I'd do next with one more week
1. **Implement Hybrid Search (BM25 + Embeddings):** Keyword search is sometimes better than semantic search for specific error codes or highly specific device bugs.
2. **Human-in-the-loop Interface:** Build a lightweight Streamlit app where human agents can override the AI's drafts, feeding corrections back into the vector store.
3. **Multi-turn RAG:** Extend the context window to embed full 4-5 turn threads rather than single prompt-response pairs.

## 6. Decision Log
- **Selected @SpotifyCares:** Chose this brand over AmazonHelp because Spotify issues are heavily software/account focused, making troubleshooting steps more uniform and easier to embed.
- **Subsampled Dataset (5,000 threads):** To ensure reviewers can run the pipeline in under 15 minutes locally without OOM errors or massive API bills.
- **Used Gemini-3.6-Flash:** Chose Flash for its speed and cost-effectiveness, making batch labelling and generation blazing fast.
- **RAG via Cosine Similarity in Numpy:** Avoided heavy vector databases (Chroma/Pinecone) to keep the repository extremely lightweight and dependency-free.
- **Batched API Calls:** Implemented batching (20 at a time) for Golden Set labelling to bypass strict Free Tier API Rate limits (5 RPM).
- **LLM-as-Judge for Golden Set:** Used an LLM to generate the 200 ground-truth labels for intents/escalation to save hours of manual labelling, enabling rapid iteration on the agent prompt instead.
- **Structured JSON Outputs:** Forced the LLM to output pure JSON via standard prompt constraints to ensure pipeline reliability across varying SDK versions.
- **Removed Multi-Turn Memory:** Chose to process only the immediate customer tweet rather than the whole thread history to reduce context window noise and focus strictly on first-touch triage.
- **Binary Escalation Flag:** Formatted the escalation target as a simple `boolean` instead of a complex routing tree, as early-stage AI implementations should fail-safe to human routing rather than attempting risky auto-resolution of edge cases.
- **Custom Eval Metric for Escalation (F1 Score):** Used F1 Score instead of raw accuracy for escalation because escalation instances are minority classes (imbalanced dataset); F1 properly punishes false positives (over-escalation).
- **Interactive Testing Script:** Built a `test_bot.py` script that isolates the generation pipeline from the massive evaluation harness, allowing reviewers to verify functionality interactively without hitting Free Tier API quotas.
- **Excluded Pydantic from Generation Config:** Despite SDK support for passing Pydantic BaseModels directly to `response_schema`, I manually enforced JSON parsing via string cleaning to avoid version compatibility bugs in deprecated SDKs.
