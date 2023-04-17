import pandas as pd

# Crear DataFrame de resultados
results_df = pd.DataFrame({'valor_real': test_labels, 'valor_predicho': predictions})

# Guardar DataFrame en un archivo CSV
results_df.to_csv('resultados.csv', index=False)
