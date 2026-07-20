# Vector Databases

A vector database stores high-dimensional embedding vectors and supports
fast nearest-neighbor search over them. Given a query vector, it returns
the stored vectors most similar to it, usually by cosine similarity or
Euclidean distance. This is the retrieval backbone of semantic search and
retrieval-augmented generation.

Exact nearest-neighbor search is linear in the number of vectors, which is
too slow at scale, so vector databases use approximate nearest-neighbor
(ANN) indexes such as HNSW or IVF to trade a little recall for large speed
gains. Popular systems include FAISS, Chroma, and pgvector. The core idea
is always the same: embed content into vectors, index them, and search by
similarity.
