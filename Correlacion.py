import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Lectura del archivo CSV
PDatos = pd.read_csv('/Users/jacquelinecarrillo/Desktop/8vo Semestre/BigData/Proyecto/new.csv', delimiter=';')

# Obtener las marcas únicas y asignarles un número único
marca_mapping = {marca: i for i, marca in enumerate(PDatos['Marca'].unique())}

# Remplazar las marcas con su número único
PDatos['Marca_Num'] = PDatos['Marca'].map(marca_mapping)

# Filtrar solo las columnas relevantes para análisis de correlación
PDatos_relevantes = PDatos[['Precio', 'Marca_Num', 'Stock']]

# Iterar sobre las marcas únicas
for marca in marca_mapping:
    # Filtrar los datos por marca
    PDatos_marca = PDatos_relevantes[PDatos_relevantes['Marca_Num'] == marca_mapping[marca]]
    
    # Correlación entre las características para esta marca
    cor_marca = PDatos_marca.corr()
    print(f"Correlación entre Precio, Marca, Stock para la marca {marca}:")
    print(cor_marca)

    # Valor máximo de correlación para esta marca
    max_corr = cor_marca.loc['Precio', ['Stock']].max()
    print(f"Valor máximo de correlación entre Precio, Modelo y Stock para la marca {marca}: {max_corr}")

    # Graficando con diagramas de dispersión todas las relaciones entre las variables para esta marca
    sns.pairplot(PDatos_marca, hue='Marca_Num')
    plt.title(f'Diagrama de dispersión para la marca {marca}')
    plt.show()

# Restaurar la marca a su forma original para la imagen
PDatos['Marca'] = PDatos['Marca_Num'].map({v: k for k, v in marca_mapping.items()})
PDatos.drop(columns=['Marca_Num'], inplace=True)
