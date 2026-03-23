from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# load data
data = load_breast_cancer()
X = data.data
y = data.target

# same split as the tree questions
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# scale features so no single column dominates just because its numbers are bigger
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# one hidden layer (two relu layers here), sigmoid at the end for binary class probability
model = Sequential()
model.add(Dense(16, activation="relu", input_shape=(X_train_scaled.shape[1],)))
model.add(Dense(8, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train_scaled, y_train, epochs=20, verbose=0)

# turn probabilities into 0/1 labels to match y
train_probs = model.predict(X_train_scaled, verbose=0)
test_probs = model.predict(X_test_scaled, verbose=0)
y_train_pred = (train_probs.ravel() > 0.5).astype(int)
y_test_pred = (test_probs.ravel() > 0.5).astype(int)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("train accuracy:", round(train_acc, 4))
print("test accuracy:", round(test_acc, 4))

# neural nets use weighted sums and gradients; if one feature is on a huge scale, updates get messy. scaling puts inputs on a similar scale.
# one epoch means the optimizer has looked at every training row once (one full pass through the training set).
