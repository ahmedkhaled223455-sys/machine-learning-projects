# K-Means Clustering

K-means partitions data into k clusters by alternating two steps: assign
each point to the nearest cluster centroid, then recompute each centroid
as the mean of its assigned points. This repeats until assignments stop
changing.

K-means minimizes within-cluster squared distance. It requires choosing k
in advance and can converge to poor local optima, so it is usually run
several times from different initializations, such as k-means++, and the
best result is kept.
