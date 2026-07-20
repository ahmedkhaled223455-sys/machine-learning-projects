# Logistic Regression

Logistic regression is a linear model for binary classification. It
computes a weighted sum of the input features and passes it through the
sigmoid function to produce a probability between zero and one. The
weights are learned by minimizing the log-loss, typically with gradient
descent.

Despite its simplicity, logistic regression is a strong baseline. Its
coefficients are interpretable: each one indicates how the log-odds of the
positive class change with a unit increase in that feature. Adding L2
regularization shrinks the weights and reduces overfitting.
