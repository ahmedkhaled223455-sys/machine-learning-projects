# Gradient Boosting

Gradient boosting builds an ensemble of decision trees sequentially, where
each new tree is fit to the residual errors of the current ensemble. By
taking small steps in the direction that reduces the loss gradient, the
model gradually improves its predictions.

Because each tree corrects the mistakes of the previous ones, gradient
boosting can achieve high accuracy. However, on small datasets its high
capacity can overfit: the model starts memorizing noise rather than
signal. Regularization through a small learning rate, shallow trees, and
subsampling helps control this. On tabular data with limited rows, a
simpler linear model sometimes generalizes better than a boosted ensemble.
