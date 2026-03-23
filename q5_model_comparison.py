import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# split data (80% train, 20% test, same class mix in both parts)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# constrained decision tree (limits depth so it does not overfit as much)
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)
tree_preds = tree_model.predict(X_test)
tree_test_acc = accuracy_score(y_test, tree_preds)

# confusion matrix for tree
tree_cm = confusion_matrix(y_test, tree_preds)
disp1 = ConfusionMatrixDisplay(
    confusion_matrix=tree_cm, display_labels=data.target_names
)
disp1.plot()
plt.title("decision tree confusion matrix")
plt.show()

print("decision tree test accuracy:", round(tree_test_acc, 4))

# neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = Sequential()
nn_model.add(Dense(16, activation="relu", input_shape=(30,)))
nn_model.add(Dense(8, activation="relu"))
nn_model.add(Dense(1, activation="sigmoid"))

nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
nn_model.fit(X_train_scaled, y_train, epochs=20, verbose=0)

# get predictions (values between 0 and 1; 0.5 splits the two classes)
nn_probs = nn_model.predict(X_test_scaled, verbose=0)
nn_preds = (nn_probs.ravel() > 0.5).astype(int)
nn_test_acc = accuracy_score(y_test, nn_preds)

# confusion matrix for neural network
nn_cm = confusion_matrix(y_test, nn_preds)
disp2 = ConfusionMatrixDisplay(
    confusion_matrix=nn_cm, display_labels=data.target_names
)
disp2.plot()
plt.title("neural network confusion matrix")
plt.show()

print("neural network test accuracy:", round(nn_test_acc, 4))

# the better model here is whichever test accuracy above is higher (compare the two printed lines).
# if the neural network number is larger, the neural network is better on this test set; if the tree number is larger, the tree is better.

# decision tree: good because rules are easy to read. bad because it can still overfit or miss subtle patterns.

# neural network: good because it can learn smooth, complex boundaries. bad because it is a "black box" and needs scaled inputs.
