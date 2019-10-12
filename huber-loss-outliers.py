'''
  Generate a BoxPlot image to determine how many outliers are within the Boston Housing Pricing Dataset.
'''
import keras
from keras.datasets import boston_housing
import numpy as np
import matplotlib.pyplot as plt

# Load the data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# We only need the targets, but do need to consider all of them
y = np.concatenate((y_train, y_test))

# Generate box plot
plt.boxplot(y)
plt.title('Boston housing price regression dataset - boxplot')
plt.show()