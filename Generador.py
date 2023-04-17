import numpy as np
import pandas as pd

# Generar datos aleatorios
n_samples = 5000 #100k de entrenamiento y aumentar la info de entrenamiento
sizes = np.random.normal(2000, 500, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
age = np.random.normal(10, 5, n_samples)
location = np.random.choice(['A', 'B', 'C', 'D'], size=n_samples)
value = 100000 + 150 * sizes + 5000 * bedrooms + 7500 * bathrooms - 2500 * age

# Redondear los valores de size a enteros y eliminar los valores menores o iguales a 3
sizes = np.round(sizes).astype(int)
sizes = sizes[sizes > 3]

# Convertir bedrooms y bathrooms a enteros
bedrooms = bedrooms.astype(int)
bathrooms = bathrooms.astype(int)

# Crear el DataFrame
data = pd.DataFrame({'size': sizes, 'bedrooms': bedrooms, 'bathrooms': bathrooms, 'age': age, 'ubicacion': location, 'valor': value})

# Guardar los datos en un archivo CSV
data.to_csv('datos_casas.csv', index=False)