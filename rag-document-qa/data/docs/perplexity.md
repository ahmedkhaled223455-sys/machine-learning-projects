# Perplexity

Perplexity measures how well a language model predicts a sample of text.
It is the exponential of the average negative log-likelihood the model
assigns to the tokens: a lower perplexity means the model found the text
less surprising and therefore modeled it better.

Perplexity is used to compare language models on held-out data and to
filter low-quality text: passages a reference model finds highly
surprising are often gibberish or corrupted, so a perplexity threshold can
screen them out of a training corpus.
