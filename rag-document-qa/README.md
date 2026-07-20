# RAG Document QA

A retrieval-augmented generation system built from the ground up:
**chunking → embeddings → vector store → similarity search → grounded
generation with citations**, plus an eval harness that measures the
retrieval half directly.

The point is to show the moving parts of RAG explicitly — what a vector
database actually does, why chunking and grounding matter — rather than
gluing together framework calls. The retrieval pipeline is pure NumPy, so
its quality is measured with real, reproducible numbers; generation uses
the same injectable-LLM pattern as the [agent project](../llm-agent-data-analyst/)
(mock for tests, real Claude when you supply a key).

## Pipeline

| Stage | File | What it does |
|---|---|---|
| Chunking | `src/chunking.py` | Splits documents into overlapping word windows so a retrieved passage is focused but keeps context. |
| Embedding | `src/embedder.py` | From-scratch **TF-IDF** vectorizer (NumPy): fits a vocabulary + IDF weights, emits L2-normalized vectors so cosine similarity is a dot product. `Embedder` interface swaps in a dense API embedder. |
| Vector store | `src/vector_store.py` | In-memory **cosine k-NN index** with save/load — the core of a vector DB, written out. Maps to FAISS/Chroma/pgvector (which add ANN indexes for scale). |
| Retrieval | `src/retriever.py` | Builds the index from the corpus; returns top-k chunks for a query. |
| Generation | `src/rag.py` | Prompts the LLM with numbered context passages and **grounding instructions** (answer only from context, cite `[n]`, else say "I don't know"), and returns the answer with its sources. |

## Grounding is the point

The generation prompt hands the model only the retrieved passages and
tells it to answer from them, cite them, and refuse when the answer isn't
present. That instruction is the main defense against hallucination in
RAG — the retrieved context is declared the only source of truth, and
every answer comes back with the source chunks it used, so it's
auditable.

## Retrieval evaluation (real, local, no LLM)

`evals/eval_questions.jsonl` pairs 12 questions with the document that
actually contains each answer. Because retrieval needs no LLM, its
quality is measured directly as **recall@k** — how often the gold
document lands in the top-k retrieved chunks:

```
$ python evals/eval_rag.py
Retrieval over 12 questions:
  recall@1: 100%
  recall@3: 100%
```

This is a genuine, reproducible result (run it yourself — no key needed).
Honest caveat: the corpus is deliberately distinct-topic technical notes,
which plays to TF-IDF's strength (distinctive keywords). Paraphrased
questions that share no vocabulary with the source are exactly where
you'd switch the `Embedder` to dense semantic embeddings — the interface
is built for that swap.

## Answer evaluation (needs a key)

With `ANTHROPIC_API_KEY` set, `--answers` also runs grounded generation
against real Claude and checks each answer contains the expected fact and
cites a source:

```bash
python evals/eval_rag.py --answers
```

## Run

```bash
pip install -r requirements.txt

# retrieval quality — no key needed
python evals/eval_rag.py

# inspect what gets retrieved for a query
python src/retriever.py "Why can gradient boosting overfit small datasets?"

# full RAG answer against real Claude
export ANTHROPIC_API_KEY=sk-ant-...
python src/rag.py "What problem do vector databases solve?"
```

## Test

```bash
python -m pytest tests/ -v
```

Covers chunking (windows + overlap), the TF-IDF embedder (normalization,
IDF down-weighting, orthogonality), the vector store (nearest-neighbor
ordering + save/load), retrieval recall on the eval set, and the RAG
assembly with a mock LLM (real retrieval feeds a grounded prompt and
sources are returned). No API key required — generation is mock-tested.

## Limitations / next steps

- TF-IDF is lexical; semantic/paraphrase queries want a dense embedder
  (Voyage, OpenAI, or local sentence-transformers) behind the same
  `Embedder` interface.
- Brute-force cosine scan is O(n); at scale swap the store for an ANN
  index (HNSW/IVF via FAISS or a managed vector DB).
- No reranking step; a cross-encoder reranker over the top-k would lift
  precision on harder corpora.
- Answer grading is keyword + citation presence; open-ended answers would
  want an LLM-as-judge rubric with its own baseline.
