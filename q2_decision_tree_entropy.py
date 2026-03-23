from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load data
data = load_breast_cancer()
X = data.data
y = data.target

# 80% train, 20% test, same class mix in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# decision tree split rule uses entropy (information gain)
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("train accuracy:", round(train_acc, 4))
print("test accuracy:", round(test_acc, 4))

# entropy here means "how mixed the labels are" in a group of rows. the tree picks splits that lower that mix.
# if train score is a lot higher than test score, the tree may be overfitting (memorizing noise). if they are close, generalization looks okay.
