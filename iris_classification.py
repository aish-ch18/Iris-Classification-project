import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build and Train the Model using Scikit-learn
model_sklearn = RandomForestClassifier(n_estimators=100, random_state=42)
model_sklearn.fit(X_train, y_train)
y_pred_sklearn = model_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"Scikit-learn Model Accuracy: {accuracy_sklearn:.2f}")

# Build and Train the Model using TensorFlow
model_tf = Sequential()
model_tf.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model_tf.add(Dense(64, activation='relu'))
model_tf.add(Dense(3, activation='softmax'))
model_tf.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model_tf.fit(X_train, y_train, epochs=50, batch_size=5, validation_split=0.2)
y_pred_tf = model_tf.predict(X_test)
y_pred_tf_classes = np.argmax(y_pred_tf, axis=1)
accuracy_tf = accuracy_score(y_test, y_pred_tf_classes)
print(f"TensorFlow Model Accuracy: {accuracy_tf:.2f}")

# Visualize the Training Process
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
