# Attention and Transformers

The attention mechanism lets a model weigh the relevance of different
input tokens when computing a representation for each position. For every
token, attention produces query, key, and value vectors; the output is a
weighted sum of values where the weights come from the similarity between
the query and each key.

Transformers stack many self-attention layers, allowing every token to
attend to every other token in the sequence. This lets the model capture
long-range dependencies in parallel, which is why transformers replaced
recurrent networks for most language tasks.
