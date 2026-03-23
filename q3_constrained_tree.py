from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# load data
data = load_breast_cancer()
X = data.data
y = data.target

# same split as other questions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# simpler tree: cap depth so it cannot grow huge rules that hug every training row
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("train accuracy:", round(train_acc, 4))
print("test accuracy:", round(test_acc, 4))

# top five features by importance (bigger number = used more in splits)
importances = model.feature_importances_
features = data.feature_names
pairs = list(zip(features, importances))
pairs.sort(key=lambda x: x[1], reverse=True)

print("\ntop 5 features:")
for name, score in pairs[:5]:
    print(name, ":", round(float(score), 4))

# adding limits like max_depth cuts model complexity, which usually lowers overfitting risk versus a full deep tree.
# importance scores help explain the model: you see which inputs drive the yes/no splits.
