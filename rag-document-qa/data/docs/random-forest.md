# Random Forest

A random forest is an ensemble of decision trees trained on bootstrap
samples of the data, with each split considering only a random subset of
features. Predictions are averaged across trees, which reduces variance
compared to a single deep tree.

The randomness decorrelates the trees so their errors cancel out when
averaged. Random forests are robust, need little tuning, and provide
feature-importance estimates, but they are larger and slower to evaluate
than a single tree and less interpretable.
