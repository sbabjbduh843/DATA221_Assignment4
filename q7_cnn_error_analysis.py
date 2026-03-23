import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# class names (same order as fashion mnist labels 0..9)
class_names = [
    "t-shirt/top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot",
]

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# keep raw test images for plotting (before reshape)
X_test_images = X_test.copy()

# normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# reshape with channel
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# same cnn as question 6
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(X_train, y_train, epochs=15, verbose=1)

# predict on test set
y_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_probs, axis=1)

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("cnn confusion matrix")
plt.tight_layout()
plt.show()

# indexes where the model was wrong
wrong_indexes = np.where(y_pred != y_test)[0]
n_show = min(3, len(wrong_indexes))

if n_show == 0:
    print("no wrong predictions on the test set (rare).")
else:
    for i in range(n_show):
        index = int(wrong_indexes[i])
        plt.imshow(X_test_images[index], cmap="gray")
        plt.title(
            "true: "
            + class_names[y_test[index]]
            + " | predicted: "
            + class_names[y_pred[index]]
        )
        plt.axis("off")
        plt.show()

# one pattern: tops like shirt, t-shirt, and pullover often get swapped because they look alike from far away.

# one way to improve: train longer, add another conv block, or use data aug (small shifts/rotations) so the net sees more variety.
