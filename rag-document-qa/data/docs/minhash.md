# MinHash and Near-Duplicate Detection

MinHash is a technique for estimating the Jaccard similarity between two
sets efficiently. It works by applying many independent hash functions to
the elements of a set and keeping only the minimum hash value from each.
The fraction of hash functions for which two sets share the same minimum
value is an unbiased estimator of their Jaccard similarity.

For near-duplicate document detection, a document is represented as a set
of character or word shingles. Comparing every pair of documents directly
is quadratic, so MinHash signatures are combined with locality-sensitive
hashing (LSH): signatures are split into bands, and only documents that
collide in at least one band are compared. This turns an intractable
all-pairs comparison into a scalable candidate-generation step.
