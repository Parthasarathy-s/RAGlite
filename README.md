# RAGlite

A tiny RAG (Retrieval-Augmented Generation) app in one file, built for learning how LLM apps work.

An LLM only knows what it was trained on. RAG lets it answer from **your** documents by looking up relevant text first and pasting it into the prompt.

## Files

- `rag.py`: the whole app (about 100 lines)
- `docs/*.txt`: the documents it answers from (add your own)

## Run

```
python3 rag.py "What is an embedding?"
export ANTHROPIC_API_KEY=...   # optional; enables the final LLM step
```

Without a key, the script stops before the LLM call and prints the exact prompt it would have sent. With a key, it also prints Claude's answer (model: `claude-haiku-5-5`, set in `generate()`).

Requires Python 3 and `pip install anthropic` (only for the LLM step).

## The six steps

| # | Step | What happens | Function |
|---|------|--------------|----------|
| 1 | Load | read text files from `docs/` | `load_docs` |
| 2 | Chunk | split text into small pieces (prompts have limited space) | `chunk` |
| 3 | Embed | turn each chunk into numbers so chunks can be compared | `embed`, `cosine` |
| 4 | Retrieve | embed the question, pick the most similar chunks | `retrieve` |
| 5 | Prompt | paste chunks + question into one prompt | `build_prompt` |
| 6 | Generate | send the prompt to the LLM, which answers from the chunks | `generate` |

## Code walkthrough

Python one-liners (comprehensions) are shown next to the plain loops they mean.

### 1. Load
```python
return {p.name: p.read_text() for p in sorted(Path(folder).glob("*.txt"))}
```
Long way:
```python
docs = {}
for p in sorted(Path(folder).glob("*.txt")):   # each .txt file in docs/
    docs[p.name] = p.read_text()               # filename -> file contents
return docs
```
Result: `{"pets.txt": "Our office dog...", "llm_basics.txt": "..."}`.

### 2. Chunk
```python
return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
```
Long way:
```python
chunks = []
for i in range(0, len(words), size):   # i = 0, 40, 80, ...
    piece = words[i:i + size]          # slice: 40 words starting at i
    chunks.append(" ".join(piece))     # glue the words back into a string
return chunks
```

### 3. Embed
```python
words = re.findall(r"[a-z0-9]+", text.lower())
return Counter(w for w in words if w not in STOPWORDS)
```
Long way:
```python
words = re.findall(r"[a-z0-9]+", text.lower())  # "Hi, Bob!" -> ["hi", "bob"]
counts = {}
for w in words:
    if w not in STOPWORDS:                      # skip filler words
        counts[w] = counts.get(w, 0) + 1        # count each word
return counts
```
The "vector" is a word count like `{"dog": 1, "biscuit": 2}`. Real systems use a neural embedding model that captures *meaning*; this one only matches shared **words** (so "hoover" will not match "vacuum"). Swapping in a real model means replacing only `embed()`.

**Stopwords** (`is`, `the`, `what`, ...) are dropped because they appear everywhere and would make unrelated chunks look similar.

### 3b. Cosine similarity
Scores two word-count dicts from 0 (nothing shared) to 1 (identical).
```python
dot = sum(a[w] * b[w] for w in a if w in b)
```
Long way:
```python
dot = 0
for w in a:
    if w in b:                  # only words both texts contain
        dot += a[w] * b[w]
```
Each text's overall "length" is `sqrt(sum of squared counts)`. Dividing by both lengths stops long chunks winning just for being long.

### 4. Retrieve
```python
scored = [(cosine(q, vec), src, text) for src, text, vec in index]
return sorted(scored, reverse=True)[:k]
```
Long way:
```python
scored = []
for src, text, vec in index:                  # every chunk
    score = cosine(q, vec)                    # compare to the question
    scored.append((score, src, text))
scored.sort(reverse=True)                     # highest score first
return scored[:k]                             # keep the top k
```
The score is first in the tuple because sorting tuples sorts by the first item.

### 5. Prompt
```python
context = "\n\n".join(f"[{src}] {text}" for _, src, text in hits)
```
Long way:
```python
parts = []
for score, src, text in hits:
    parts.append(f"[{src}] {text}")   # f-string fills {src} and {text} in
context = "\n\n".join(parts)          # blank line between chunks
```
`_` is the usual name for a value you do not use (here, the score). The prompt tells the model to answer **only** from the context, and to say it does not know otherwise.

### 6. Generate
Sends the prompt to the Anthropic API. `msg.content[0].text` is the first block of the reply, which is the answer text.

### The index in `main()`
```python
index = [(src, c, embed(c)) for src, text in load_docs().items() for c in chunk(text)]
```
Long way:
```python
index = []
for src, text in load_docs().items():     # each file
    for c in chunk(text):                 # each chunk in that file
        index.append((src, c, embed(c)))  # remember: file, chunk, its vector
```
The index is this app's "database": a list of `(filename, chunk text, vector)` tuples.

### Reading any comprehension
`[EXPRESSION for X in THINGS if CONDITION]` means:
```python
result = []
for X in THINGS:
    if CONDITION:
        result.append(EXPRESSION)
```

## Things to try

- Change `CHUNK_WORDS` and `TOP_K` in `rag.py` and watch the answers change.
- Ask something that is not in the docs and check that it says it does not know.
- Ask "Which animal hides from the hoover?" and see keyword matching fail.
- Replace `embed()` with a real embedding model (e.g. sentence-transformers).
