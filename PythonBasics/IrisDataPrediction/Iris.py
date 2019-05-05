import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# DataSet from file by pandas
dataSet = pd.read_csv('Models/iris.data', header=None,
                      names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Transform label data to vector who could be understand by algorithm
species_lb = LabelBinarizer()
Y = species_lb.fit_transform(dataSet.species.values)

# Normalize data from features
FEATURES = dataSet.columns[0:4]
X_data = dataSet[FEATURES].values
X_data = normalize(X_data)

# Spiting data to train and test data
X_train, X_test, y_train, y_test = train_test_split(dataSet[FEATURES].values, Y, test_size=0.3, random_state=1)

print('Inputs: {:d}\nOutputs: {:d}'.format(X_train.shape[1], y_train.shape[1]))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=[X_train.shape[1]]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation=tf.nn.sigmoid))

# region Compiling model
model.compile(optimizer='RMSprop', loss='mean_squared_error')
# endregion

# Fit
model.fit(X_train, y_train, epochs=2000, batch_size=32)

# Test
score = model.evaluate(X_test, y_test)
print(score)
print('Test loss: {0:.2f}%'.format(score*100))

# region Predict
testValSetosa = np.array([[4.4, 2.9, 1.4, 0.2]])
testValVersicolor = np.array([[6.7, 3.1, 4.4, 1.4]])
testValVirginica = np.array([[6.2, 2.8, 4.8, 1.8]])

predictSetosa = model.predict(testValSetosa)
predictVersicolor = model.predict(testValVersicolor)
predictVirginica = model.predict(testValVirginica)

print(predictSetosa)
print(predictVersicolor)
print(predictVirginica)
# endregion
