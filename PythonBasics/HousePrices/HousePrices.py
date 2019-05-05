import tensorflow as tf
from PythonBasics.HousePrices.Models.XYData import XYData

model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(1, activation=tf.nn.relu, input_shape=[1])]
)

model.compile(optimizer='sgd',
              loss='mean_squared_error')
data = XYData()

model.fit(data.inputData, data.inputLabel, epochs=2000, batch_size=32, )
result = model.predict(data.inputPredict)
print(result)
