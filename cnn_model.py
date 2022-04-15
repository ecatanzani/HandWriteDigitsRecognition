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
filters = 64
dropout = 0.3

# using functional API to build cnn layers
inputs = Input(shape=input_shape)
conv2d_l1 = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(inputs)
maxpool2d_l1 = MaxPooling2D()(conv2d_l1)
conv2d_l2 = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(maxpool2d_l1)
maxpool2d_l2 = MaxPooling2D()(conv2d_l2)
conv2d_l3 = Conv2D(filters=filters, kernel_size=kernel_size, activation='relu')(maxpool2d_l2)
# convert image to vector 
flat = Flatten()(conv2d_l3)
# dropout regularization
drop = Dropout(dropout)(flat)
outputs = Dense(num_labels, activation='softmax')(drop)
# model building by supplying inputs/outputs
model = Model(inputs=inputs, outputs=outputs)

# Plot the model
font = ImageFont.truetype("arial.ttf", 32)
visualkeras.layered_view(model, to_file='model_layered_view.png', legend=True)

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model
model_history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=batch_size)

# Plot the categorical cross-entropy in the epochs
plt.plot (model_history.history['loss'], 'o-', label="Training")
plt.plot (model_history.history['val_loss'], 'o-', label="Validation")

plt.yscale('log')
plt.xlabel ("Epoch")
plt.ylabel ("Categorical cross-entropy")
plt.legend()
plt.show()

score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=False)

print(f"Loss value: {score[0]} --> Model score: {score[1]*100} %")

# Save model to disk
model.save("model_cnn.h5")