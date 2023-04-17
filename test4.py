import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv("C:\laragon\www\Redes Neuronales\Prediccion-de-casas\datos_casas.csv")

# Preprocesamiento de los datos
data = data.dropna()

# Normalización de los datos
data = (data - data.mean(numeric_only=True)) / data.std(numeric_only=True)

# Codificación de variables categóricas
data = pd.get_dummies(data, columns=["ubicacion"])

# Dividir los datos en entrenamiento y prueba
train_data = data.sample(frac=0.8, random_state=0)
test_data = data.drop(train_data.index)

# Imprimir valores pasados y futuros de algunas casas
print("Valores de casas pasados y futuros:")
print(test_data.head())

# Separar las características y la etiqueta
train_labels = train_data.pop('valor')
test_labels = test_data.pop('valor')

# Construir el modelo
model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_data.keys())]),
    keras.layers.Dense(32, activation=tf.nn.relu),#puede cambiarse a una lineal
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(8, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.02)),
    keras.layers.Dense(1)
])

# Compilar el modelo
opt = keras.optimizers.Adam(lr=0.01)
model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse'])

# Entrenar el modelo
history = model.fit(train_data, train_labels, epochs=500, validation_split=0.5, verbose=0)

# Evaluar el modelo en el conjunto de prueba
loss, mae, mse = model.evaluate(test_data, test_labels, verbose=0)

# Hacer predicciones con el modelo
predictions = model.predict(test_data).flatten()

print("Mean Squared Error:", mse)

# Gráfica de la función de pérdida durante el entrenamiento
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Función de pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Gráfica de comparación entre los valores predichos y reales en el conjunto de prueba
plt.scatter(test_labels, predictions)
plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'r--')
plt.title('Valores predichos vs. reales en el conjunto de prueba')
plt.xlabel('Valor real')
plt.ylabel('Valor predicho')
plt.show()