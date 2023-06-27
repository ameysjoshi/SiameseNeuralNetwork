import numpy as np
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Lambda, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape the input images
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


# Define the Siamese network architecture
def siamese_network():
    # Input placeholders
    input_a = Input(shape=(28, 28, 1))
    input_b = Input(shape=(28, 28, 1))

    # Shared convolutional layers
    convnet = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='sigmoid')
    ])

    # Apply shared layers to both inputs
    encoded_a = convnet(input_a)
    encoded_b = convnet(input_b)

    # Compute Euclidean distance between the encoded inputs
    distance = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)))([encoded_a, encoded_b])

    # Define the Siamese model
    siamese_model = Model(inputs=[input_a, input_b], outputs=distance)

    return siamese_model


# Contrastive loss function
def contrastive_loss(y_true, y_pred):
    margin = 1
    y_true = K.cast(y_true, dtype='float32')  # Cast y_true to float32
    return K.mean((1 - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0)))



# Compile the model
model = siamese_network()
model.compile(loss=contrastive_loss, optimizer=RMSprop(), metrics=['accuracy'])


# Generate training pairs
def generate_pairs(x, y, num_pairs):
    pairs = []
    labels = []
    digits = np.unique(y)

    for _ in range(num_pairs):
        # Select a random digit
        digit = np.random.choice(digits)

        # Select two random samples for the same digit
        indices = np.where(y == digit)[0]
        idx1, idx2 = np.random.choice(indices, size=2, replace=False)
        pairs.append([x[idx1], x[idx2]])
        labels.append(1)

        # Select two random samples for different digits
        indices = np.where(y != digit)[0]
        idx1, idx2 = np.random.choice(indices, size=2, replace=False)
        pairs.append([x[idx1], x[idx2]])
        labels.append(0)

    return np.array(pairs), np.array(labels)


# Generate training and validation pairs
train_pairs, train_labels = generate_pairs(x_train, y_train, num_pairs=50000)
val_pairs, val_labels = generate_pairs(x_test, y_test, num_pairs=10000)

# Train the model
model.fit([train_pairs[:, 0], train_pairs[:, 1]], train_labels,
          validation_data=([val_pairs[:, 0], val_pairs[:, 1]], val_labels),
          batch_size=128,
          epochs=30)