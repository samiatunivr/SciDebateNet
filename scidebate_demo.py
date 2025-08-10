"""
scidebate_demo.py
Enhanced multi-agent debate pipeline with OpenAI integration, strict prompting, logging, and basic JSON outputs.

Usage:
  - Place .txt files (papers/abstracts) in ./papers/
  - Set OPENAI_API_KEY env var to enable real LLM usage
  - python scidebate_demo.py

For Colab: run the same file after installing requirements.
"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

# optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# FAISS
import faiss

# Config
DOCS_PATH = Path("./papers")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
N_RETRIEVE = 5
DEBATE_ROUNDS = 2
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY")) and OPENAI_AVAILABLE
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Utilities
def load_docs(folder: Path) -> List[Dict[str,str]]:
    docs = []
    if not folder.exists():
        raise FileNotFoundError(f"{folder} not found. Create it and add .txt files (paper abstracts).")
    for f in sorted(folder.glob("*.txt")):
        text = f.read_text(encoding="utf-8")
        docs.append({"id": str(f.name), "text": text[:20000]})
    return docs

class Retriever:
    def __init__(self, embedding_model_name=EMBEDDING_MODEL):
        self.embedder = SentenceTransformer(embedding_model_name)
        self.index = None
        self.texts = []
        self.ids = []

    def build(self, docs: List[Dict[str,str]]):
        self.texts = [d["text"] for d in docs]
        self.ids = [d["id"] for d in docs]
        embeddings = self.embedder.encode(self.texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        # use inner product on normalized vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.embeddings = embeddings

    def query(self, query_text: str, topk=5):
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, topk)
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({"id": self.ids[idx], "text": self.texts[idx], "score": float(score)})
        return results

class LLM:
    def __init__(self, use_openai=USE_OPENAI, model_name=OPENAI_MODEL):
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.model_name = model_name
        self.stub_mode = not self.use_openai
        if self.use_openai:
            openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate(self, messages: List[Dict[str,str]], max_tokens=512, temperature=0.2) -> str:
        # messages: list of dicts {role: system/user/assistant, content:...}
        if self.stub_mode:
            # Simple deterministic stub based on last user content
            user_msg = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
            stub = "Hypothesis: (stub) Drug A modulates cytokine pathway B leading to decreased inflammation in condition C. Rationale: mechanistic plausibility."
            return stub
        else:
            resp = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp["choices"][0]["message"]["content"].strip()

# Agent
class Agent:
    def __init__(self, name: str, role_system_prompt: str, llm: LLM, retriever: Retriever):
        self.name = name
        self.role_system_prompt = role_system_prompt
        self.llm = llm
        self.retriever = retriever

    def act(self, user_context: str, evidence: List[Dict[str,Any]]=None) -> str:
        evidence_text = ""
        if evidence:
            evidence_text = "\n\nEvidence (top results):\n" + "\n".join([f"- [{e['id']}] (score={e['score']:.3f})\n{e['text'][:600].replace('\n',' ')}" for e in evidence])
        messages = [
            {"role":"system","content": self.role_system_prompt},
            {"role":"user","content": f"Context:\n{user_context}\n\n{evidence_text}\n\nPlease respond concisely."}
        ]
        out = self.llm.generate(messages)
        return out

# Prompts
GEN_PROMPT = (
    "You are the Generator agent. Propose 1 concise, testable scientific hypothesis based on the context. "
    "Include: hypothesis statement, short rationale (2-3 lines), and 1-2 measurable predictions or experiments to test it. "
    "Be explicit about variables and mechanisms."
)
CRITIC_PROMPT = (
    "You are the Critic agent. For the provided hypothesis, list up to 5 weaknesses, possible confounders or counter-evidence, and suggest experiments or queries that would invalidate the hypothesis. "
    "Be precise and cite any relevant evidence passages if present."
)
SYNTH_PROMPT = (
    "You are the Synthesizer agent. Merge the generator hypothesis and critic feedback into a single refined, testable hypothesis. "
    "List key assumptions, propose a basic experimental design (controls, endpoints), and list up to 3 supporting citations pulled from evidence."
)

# Scoring
import numpy as np

def compute_novelty(hypothesis: str, retriever: Retriever) -> float:
    emb = retriever.embedder.encode([hypothesis], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = retriever.index.search(emb, 5)
    mean_sim = float(np.mean(D))
    return max(0.0, 1.0 - mean_sim)

def compute_evidence_score(hypothesis: str, retriever: Retriever) -> float:
    emb = retriever.embedder.encode([hypothesis], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = retriever.index.search(emb, N_RETRIEVE)
    return float(np.max(D))

# Runner

def run_debate(seed_prompt: str, retriever: Retriever, llm: LLM, rounds=DEBATE_ROUNDS):
    gen = Agent("Generator", GEN_PROMPT, llm, retriever)
    crit = Agent("Critic", CRITIC_PROMPT, llm, retriever)
    synth = Agent("Synthesizer", SYNTH_PROMPT, llm, retriever)

    history = []
    context = seed_prompt

    for r in range(rounds):
        evidence = retriever.query(context, topk=N_RETRIEVE)
        gen_out = gen.act(context, evidence)
        history.append({"turn": f"gen_{r}", "text": gen_out, "evidence": evidence})

        # Critic uses generator output as context
        crit_evidence = retriever.query(gen_out, topk=N_RETRIEVE)
        crit_out = crit.act(gen_out, crit_evidence)
        history.append({"turn": f"crit_{r}", "text": crit_out, "evidence": crit_evidence})

        # update context to include both
        context = f"Seed: {seed_prompt}\n\nGenerator: {gen_out}\n\nCritic: {crit_out}"

    # Synthesize
    final_input = context
    synth_evidence = retriever.query(final_input, topk=N_RETRIEVE)
    synth_out = synth.act(final_input, synth_evidence)
    history.append({"turn": "synth", "text": synth_out, "evidence": synth_evidence})

    final_hyp = synth_out
    novelty = compute_novelty(final_hyp, retriever)
    evidence_score = compute_evidence_score(final_hyp, retriever)

    return {"history": history, "final_hypothesis": final_hyp, "novelty": novelty, "evidence_score": evidence_score}

# Main

def main():
    print("Loading docs...")
    docs = load_docs(DOCS_PATH)
    print(f"{len(docs)} docs loaded.")
    retr = Retriever()
    retr.build(docs)
    llm = LLM()

    seed = (
        "Explore potential drug repurposing hypotheses for treating a viral infection with cytokine storm-like symptoms. "
        "Focus on molecular mechanisms and testable interventions."
    )

    print("Running debate...")
    out = run_debate(seed, retr, llm, rounds=DEBATE_ROUNDS)
    ts = int(time.time())
    with open(f"debate_result_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Done. Results written.")

if __name__ == '__main__':
    main()
