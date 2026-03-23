import numpy as np
from sklearn.datasets import load_breast_cancer

# load dataset
data = load_breast_cancer()

# feature matrix X and target y
X = data.data
y = data.target

# shapes the assignment asks for
print("X shape:", X.shape)
print("y shape:", y.shape)

# class names (order matches labels 0, 1, ...)
print("class names:", list(data.target_names))

# how many samples per class
unique, counts = np.unique(y, return_counts=True)
for label, count in zip(unique, counts):
    name = data.target_names[label]
    print("class", label, "(" + name + "):", count, "samples")

# the classes are not equal in size (more of one label than the other), so the set is a little imbalanced.
# that matters because a model can look accurate by mostly guessing the bigger class and still do poorly on the smaller class.
