from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# load data
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# normalize pixel values to 0..1
X_train = X_train / 255.0
X_test = X_test / 255.0

# add channel dimension: height, width, channels (1 because grayscale)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# build cnn
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(10, activation="softmax"))

# compile for 10 classes
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# train at least 15 epochs (assignment asks for 15+)
model.fit(X_train, y_train, epochs=15, verbose=1)

# report test accuracy
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print("test accuracy:", round(float(acc), 4))

# cnn is usually better than a fully connected net on images because it shares weights over small patches,
# so it notices local patterns (edges, corners) and does not tie every pixel to every weight at once.

# the conv layer learns small filters that fire on simple shapes and textures; deeper layers can mix those into bigger parts.
