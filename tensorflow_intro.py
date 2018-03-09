"""TensorFlow intro"""

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 5
pd.options.display.float_format = '{:.1f}'.format

URL = "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
CALIFORNIA_HOUSING_DATAFRAME = pd.read_csv(URL, sep=",")

CALIFORNIA_HOUSING_DATAFRAME = CALIFORNIA_HOUSING_DATAFRAME.reindex(
    np.random.permutation(CALIFORNIA_HOUSING_DATAFRAME.index)
)

CALIFORNIA_HOUSING_DATAFRAME["median_house_value"] /= 1000.0
#print(CALIFORNIA_HOUSING_DATAFRAME.describe())

# Define the input feature: total_rooms.
my_feature=CALIFORNIA_HOUSING_DATAFRAME[["total_rooms"]]
#print(type(my_feature))

# Configure a numeric feature column for total_rooms.
feature_columns=[tf.feature_column.numeric_column("total_rooms")]

# Define the label
targets=CALIFORNIA_HOUSING_DATAFRAME["median_house_value"]
#print(type(targets))

# Use gradient descent as the optimizer for training the model.
my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0000001)

# Gradient clipping ensures the magnitude of the gradients do not become too large
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

# Configure the linear regression model with our feature columns and optimizer.
# Set a learning rate of 0.0000001 for Gradient Descent.
linear_regressor=tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=my_optimizer
)


