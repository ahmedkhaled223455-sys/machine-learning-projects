# TF-IDF

TF-IDF (term frequency-inverse document frequency) scores how important a
word is to a document within a collection. Term frequency rewards words
that appear often in a document, while inverse document frequency
down-weights words that appear in many documents, so common words like
"the" contribute little.

The result is a sparse vector per document that emphasizes distinctive
terms. TF-IDF vectors compared by cosine similarity give a simple, strong
baseline for keyword-based retrieval, without any learned embeddings.
