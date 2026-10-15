"""A tiny RAG (Retrieval-Augmented Generation) app, in one file.

An LLM only knows what it was trained on. RAG lets it answer from YOUR documents:

    1. LOAD      read text from docs/
    2. CHUNK     split it into small pieces (an LLM prompt has limited space)
    3. EMBED     turn each piece into a vector of numbers (so we can compare meaning)
    4. RETRIEVE  embed the question, find the chunks whose vectors are most similar
    5. PROMPT    paste those chunks + the question into one prompt
    6. GENERATE  send the prompt to the LLM; it answers using the chunks

Run:  python rag.py "your question"
Set ANTHROPIC_API_KEY to get step 6. Without it, the script stops after step 5
and prints the prompt, so you can see exactly what the LLM would receive.
"""
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

CHUNK_WORDS = 40   # size of each chunk
TOP_K = 2          # how many chunks to retrieve


# 1. LOAD ---------------------------------------------------------------
def load_docs(folder="docs"):
    return {p.name: p.read_text() for p in sorted(Path(folder).glob("*.txt"))}


# 2. CHUNK --------------------------------------------------------------
def chunk(text, size=CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


# 3. EMBED --------------------------------------------------------------
# Real systems use a neural embedding model (e.g. sentence-transformers, Voyage)
# that captures *meaning*. Here we use a word-count vector so there is nothing
# to install: it only matches shared WORDS, not synonyms. Swap this one function
# for a real model and the rest of the app stays identical.
STOPWORDS = {"a", "an", "the", "is", "are", "was", "of", "to", "and", "in", "it", "what", "who", "how", "which"}


def embed(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return Counter(w for w in words if w not in STOPWORDS)  # drop filler words that match everything


def cosine(a, b):
    dot = sum(a[w] * b[w] for w in a if w in b)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


# 4. RETRIEVE -----------------------------------------------------------
def retrieve(question, index, k=TOP_K):
    q = embed(question)
    scored = [(cosine(q, vec), src, text) for src, text, vec in index]
    return sorted(scored, reverse=True)[:k]


# 5. PROMPT -------------------------------------------------------------
def build_prompt(question, hits):
    context = "\n\n".join(f"[{src}] {text}" for _, src, text in hits)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


# 6. GENERATE -----------------------------------------------------------
def generate(prompt):
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    msg = client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def main():
    question = " ".join(sys.argv[1:]) or "What is RAG?"

    index = [(src, c, embed(c)) for src, text in load_docs().items() for c in chunk(text)]
    print(f"Indexed {len(index)} chunks.\n")

    hits = retrieve(question, index)
    print("--- RETRIEVED CHUNKS ---")
    for score, src, text in hits:
        print(f"score={score:.2f} [{src}] {text[:80]}...")

    prompt = build_prompt(question, hits)
    print("\n--- PROMPT SENT TO LLM ---\n" + prompt)

    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n--- ANSWER ---\n" + generate(prompt))
    else:
        print("\n(Set ANTHROPIC_API_KEY to see the LLM's answer.)")


if __name__ == "__main__":
    main()
