import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images = test_images.reshape(-1, 28, 28, 1) / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(8, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=3)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nTest accuracy: {test_acc:.2%}')

random_idx = np.random.randint(0, len(test_images))
plt.imshow(test_images[random_idx].reshape(28,28), cmap='gray')
prediction = model.predict(test_images[random_idx:random_idx+1])
print(f'\nTrue digit: {test_labels[random_idx]}')
print(f'Predicted digit: {prediction.argmax()} with {prediction.max():.1%} confidence')
plt.show()