import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Lectura del archivo CSV
PDatos = pd.read_csv('/Users/jacquelinecarrillo/Desktop/8vo Semestre/BigData/Proyecto/Celulares2.csv', delimiter=';')

# Obtener las marcas únicas
marcas_unicas = PDatos['Marca'].unique()

# Seleccionar solo las columnas numéricas
PDatos_numeric = PDatos.select_dtypes(include=np.number)

# Iterar sobre las marcas únicas
for marca in marcas_unicas:
    # Filtrar los datos por marca
    PDatos_marca = PDatos[PDatos['Marca'] == marca]
    
    # Seleccionar solo las columnas numéricas para esta marca
    PDatos_marca_numeric = PDatos_marca.select_dtypes(include=np.number)
    
    # Correlación entre todas la variables para esta marca
    cor_total = PDatos_marca_numeric.corr()
    print(f"Correlación entre todas las variables para la marca {marca}:")
    print(cor_total)

    # Valor máximo de correlación para esta marca
    max_corr = cor_total.values[np.triu_indices(cor_total.shape[0], k=1)].max()
    print(f"Valor máximo de correlación para la marca {marca}: {max_corr}")

    # Graficando con diagramas de dispersión todas las relaciones entre las variables para esta marca
    sns.pairplot(PDatos_marca_numeric)
    plt.title(f'Diagrama de dispersión para la marca {marca}')
    plt.show()

