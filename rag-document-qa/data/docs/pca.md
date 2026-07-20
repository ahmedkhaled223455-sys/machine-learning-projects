# Principal Component Analysis

Principal component analysis (PCA) reduces the dimensionality of data by
projecting it onto the directions of greatest variance. These directions,
the principal components, are the eigenvectors of the data's covariance
matrix, ordered by how much variance they explain.

PCA turns a set of possibly correlated features into a smaller set of
uncorrelated components while retaining most of the variation. It is used
for visualization, noise reduction, and speeding up downstream models, but
the transformed components are linear combinations and lose direct
interpretability.
