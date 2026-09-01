# Local RAG Assistant — Philosophical AI

An offline Retrieval-Augmented Generation (RAG) Q&A system that answers
questions about Albert Camus, free will, and Virginia Woolf using a local
knowledge base and a fully on-device language model — no internet
connection or cloud API required at runtime.

Built as a one-month summer project following Microsoft's guide to
building a local RAG application with **Foundry Local**.

<img width="1104" height="482" alt="image" src="https://github.com/user-attachments/assets/87532ba5-e47e-4b88-9e45-c9a63fee7412" />

<img width="803" height="730" alt="image" src="https://github.com/user-attachments/assets/df120d1a-66a2-410e-8c89-5760536af6b8" />

<img width="784" height="486" alt="image" src="https://github.com/user-attachments/assets/0ee5dbe2-a825-4c5e-b632-ee07cc15ecca" />

---

## What it does

You ask a question (via the command line or the web interface), and the
system:

1. Converts your question into an embedding vector.
2. Searches a local SQLite database of pre-processed knowledge base
   chunks for the most semantically similar ones (cosine similarity).
3. Passes the retrieved chunks, as context, to a local LLM.
4. Returns an answer that is grounded **only** in that context — if the
   knowledge base doesn't cover the question, the assistant says so
   instead of guessing.

The current knowledge base covers Camus's philosophy of the absurd,
several core positions on free will, and Virginia Woolf's writing and
life — 13 short chunks in total, sourced from the Stanford Encyclopedia
of Philosophy, Britannica, Wikipedia, and academic sources on Woolf,
paraphrased into original text.

---

## Architecture

```
User question
      │
      ▼
Embed query  ──────────────►  qwen3-embedding-0.6b (Foundry Local)
      │
      ▼
Cosine similarity search  ──► SQLite (rag.db) — stored chunks + vectors
      │
      ▼
Top-k relevant chunks
      │
      ▼
LLM generation  ───────────►  phi-3.5-mini (Foundry Local)
      │
      ▼
Answer (grounded in context, or a fallback if nothing relevant is found)
```

Everything — embedding model, chat model, and the vector search — runs
entirely on-device via Foundry Local. No data leaves the machine.

---

## Project structure

```
local-rag-assistant/
├── db.py            # SQLite setup: documents table, insert/read helpers
├── embed.py         # Ingests the knowledge base: chunks → embeddings → SQLite
├── retrieve.py       # get_top_chunks(): semantic search over the DB
├── main.py           # CLI version — ask questions in the terminal
├── app.py            # Web version — Flask backend (POST /ask)
├── templates/
│   └── index.html    # Web frontend
└── rag.db             # SQLite database (created by embed.py)
```

---

## Setup

1. Install dependencies:
   ```bash
   pip install foundry-local-sdk flask
   ```

2. Build the knowledge base (downloads the embedding model on first run,
   then embeds and stores all chunks):
   ```bash
   python embed.py
   ```

3. Run the assistant — either interface works independently:

   **Command line:**
   ```bash
   python main.py
   ```

   **Web interface:**
   ```bash
   python app.py
   ```
   Then open `http://127.0.0.1:5000` in a browser.

Both `main.py` and `app.py` download and cache the `phi-3.5-mini` chat
model on first run.

---

## Design decisions

- **SQLite over an in-memory store**: embeddings are computed once and
  persist across runs, and are shared between the ingestion script and
  the query-time scripts, which run as separate processes.
- **Fallback response instead of guessing**: if the best-matching chunk's
  similarity score falls below a threshold (`RELEVANCE_THRESHOLD = 0.4`),
  the system skips the LLM call entirely and returns a fixed fallback
  answer. This makes grounding a deterministic guarantee rather than
  something that depends on the model choosing to follow instructions.
- **System prompt allows semantic, not just literal, matching**: an
  earlier version of the prompt required the context to use the exact
  wording of the question, which caused the assistant to wrongly refuse
  to answer when the relevant information was present but phrased
  differently. The prompt was loosened to allow reasoning about meaning,
  while still forbidding any outside knowledge.
- **Web UI shows retrieved sources**: each answer displays the chunks
  used and their similarity scores, so the retrieval step is visible
  rather than a black box.

---

## Known limitations

- The knowledge base is small (13 chunks) and manually curated; it does
  not scale to large document collections without a proper vector index.
- Retrieval quality depends on `k` (chunks retrieved) and the relevance
  threshold — both were tuned by hand through testing, not derived
  systematically.
- The chat model (`phi-3.5-mini`) is small, and its exact wording can
  vary between runs even for similar questions; behavior was verified
  through manual testing rather than automated evaluation.

---

## Testing notes

The system was manually tested with three categories of questions —
answerable from the knowledge base, unrelated (e.g. "What is the capital
of France?"), and cross-topic questions designed to invite hallucination
(e.g. "What did Camus think of Virginia Woolf?"). During this process,
three issues were found and fixed:

1. **Hallucination**: the model added invented details beyond the given
   context. Fixed with a stricter system prompt and a similarity
   threshold that bypasses the LLM entirely for low-relevance queries.
2. **Retrieved chunk excluded by top-k limit**: a relevant chunk ranked
   just outside the top 2 retrieved chunks. Fixed by increasing `k`.
3. **Inconsistent fallback formatting**: the model sometimes added its
   own explanation before the fallback sentence instead of returning it
   verbatim. Fixed by making the "no outside content" instruction more
   explicit in the system prompt.
