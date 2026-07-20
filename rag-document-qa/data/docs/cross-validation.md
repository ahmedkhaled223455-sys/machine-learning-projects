# Cross-Validation

Cross-validation estimates how well a model generalizes by repeatedly
splitting the data into training and validation folds. In k-fold
cross-validation the data is divided into k parts; the model trains on
k-1 parts and validates on the held-out part, rotating until every part
has served as validation once.

Averaging the validation scores gives a more reliable performance estimate
than a single split, and it uses the data efficiently. Preprocessing such
as scaling must be fit inside each fold to avoid leaking information from
the validation set into training.
