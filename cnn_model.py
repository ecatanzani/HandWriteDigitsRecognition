import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.datasets import mnist
import visualkeras
from PIL import ImageFont

# load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# convert sparse label to categorical values
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# preprocess the input images
image_size = x_train.shape[1]

x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# parameters for the network
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3

#Define the model
def build_model(input_shape, batch_size, kernel_size, num_labels, internal_dropout, external_dropout):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(internal_dropout))
    model.add(Conv2D(filters=64, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(internal_dropout))
    model.add(Conv2D(filters=128, kernel_size=kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(internal_dropout))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(external_dropout))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(external_dropout))
    model.add(Dense(num_labels, activation='softmax'))
    return model

model = build_model(input_shape, batch_size, kernel_size, num_labels, internal_dropout=0.3, external_dropout=0.5)

# Plot the model
font = ImageFont.truetype("arial.ttf", 32)
visualkeras.layered_view(model, to_file='model_layered_view.png', legend=True)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=batch_size)

#Evaluate the network performances
plt.plot(model_history.history['loss'], label="Training")
plt.plot(model_history.history['val_loss'], label="Validation")

plt.title("Loss")
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Categorical cross-entropy")

plt.legend()
plt.show()

plt.plot(model_history.history['accuracy'], label="Training")
plt.plot(model_history.history['val_accuracy'], label="Validation")

plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0.95, 1)

plt.legend()
plt.show()

score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=False)

print(f"Loss value on test: {score[0]} --> Model score on test: {score[1]*100} %")

# Save model to disk
model.save("model_cnn.h5")