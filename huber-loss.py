'''
  Keras model demonstrating Huber loss
'''
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import huber_loss
import numpy as np
import matplotlib.pyplot as plt

# Define the Huber loss so that it can be used with Keras
def huber_loss_wrapper(**huber_loss_kwargs):
    def huber_loss_wrapped_function(y_true, y_pred):
        return huber_loss(y_true, y_pred, **huber_loss_kwargs)
    return huber_loss_wrapped_function

# Load data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# Set the input shape
shape_dimension = len(x_train[0])
input_shape = (shape_dimension,)
print(f'Feature shape: {input_shape}')

# Create the model
model = Sequential()
model.add(Dense(16, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='linear'))

# Configure the model and start training
model.compile(loss=huber_loss_wrapper(delta=1.5), optimizer='adam', metrics=['mean_absolute_error'])
history = model.fit(x_train, y_train, epochs=250, batch_size=1, verbose=1, validation_split=0.2)

# Test the model after training
test_results = model.evaluate(x_test, y_test, verbose=1)
print(f'Test results - Loss: {test_results[0]} - MAE: {test_results[1]}')

# Plot history: Huber loss and MAE
plt.plot(history.history['loss'], label='Huber loss (training data)')
plt.plot(history.history['val_loss'], label='Huber loss (validation data)')
plt.title('Boston Housing Price Dataset regression model - Huber loss')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

plt.title('Boston Housing Price Dataset regression model - MAE')
plt.plot(history.history['mean_absolute_error'], label='MAE (training data)')
plt.plot(history.history['val_mean_absolute_error'], label='MAE (validation data)')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()