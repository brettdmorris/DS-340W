# Implementation Platform (Parent Paper)
Gradient Boost and SVM are the implementations for novelty. Our initial implementation using Random Forest had data leakage from the parent paper and that is fixed in the newest versions.

Comparing to a baseline accuracy of 56% for significance in stock trend prediction, we were able to achieve a 10-day stock trend prediction accuracy of 55.17% for Random Forest, 51.46% for Gradient Boosting, and 62.07% for Support Vector Machines.

This increases with longer-term predictions as the general stock movements over time are positive.
For a 30-day prediction: RF accuracy: 61.16%, GB accuracy: 64.64%, SVM accuracy: 69.57%
For a 90-day prediction all models approach around an 89% accuracy as long-term trends are easier to predict.
