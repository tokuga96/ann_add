import numpy as np
import tensorflow as tf
from tensorflow import keras

train_data = np.array([[1.0, 1.0]])
train_targets = np.array([2.0])
print(train_data)

for i in range(3, 10000, 2):
    train_data = np.append(train_data, [[i, i]], axis=0)
    train_targets = np.append(train_targets, [i + i])
test_data = np.array([[2.0, 2.0]])
test_targets = np.array([4.0])

for i in range(4, 8000, 4):
    test_data = np.append(test_data, [[i, i]], axis=0)
    test_targets = np.append(test_targets, [i + i])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(20, activation=tf.nn.relu),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

model.fit(train_data, train_targets, epochs=10, batch_size=1)

test_loss, test_acc = model.evaluate(test_data, test_targets)
print('Test accuracy:', test_acc)
a = np.array([[2000, 3000], [4, 5]])
print(model.predict(a))
