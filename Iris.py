import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Cargar datos desde el archivo CSV
df = pd.read_csv('/Users/jacquelinecarrillo/Desktop/8vo Semestre/BigData/Proyecto/new.csv', delimiter=";")

# Copiar el DataFrame original para mantener las marcas originales
df_original = df.copy()

# Codificar las características categóricas (marca y modelo)
label_encoder = LabelEncoder()
df['Marca'] = label_encoder.fit_transform(df['Marca'])
df['Modelo'] = label_encoder.fit_transform(df['Modelo'])

# Obtener el mapeo entre los números asignados y las marcas originales
marca_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))

# Seleccionar las variables relevantes para K-means
df_cluster = df[["Precio", "Marca", "Modelo", "Stock"]]

# Normalizar las características numéricas (precio y stock)
df_cluster["Precio"] = df_cluster["Precio"] / df_cluster["Precio"].max()
df_cluster["Stock"] = df_cluster["Stock"] / df_cluster["Stock"].max()

# Convertir el DataFrame a una matriz numpy para K-means
x = df_cluster.values

# Encontrar el valor óptimo de clusters utilizando el método del codo
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

# Graficar el método del codo
plt.plot(range(1, 11), wcss)
plt.title("El método del codo")
plt.xlabel("Número de clusters")
plt.ylabel("WCSS")
plt.show()

# Aplicar K-means con el número óptimo de clusters
num_clusters = 3  # Número de clusters óptimo seleccionado
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_kmeans = kmeans.fit_predict(x)

# Agregar la columna de clusters al DataFrame original
df_original['Cluster'] = y_kmeans

# Visualizar los clusters en un gráfico de dispersión
sns.scatterplot(data=df_original, x='Precio', y='Stock', hue='Marca', style='Cluster', palette='viridis')
plt.xlabel('Precio (Normalizado)')
plt.ylabel('Stock (Normalizado)')
plt.title('Gráfico de dispersión de Precio vs Stock por Marca y Cluster')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))

# Imprimir el mapeo de números a marcas originales
print("Mapeo de números a marcas originales:")
print(marca_mapping)

plt.show()
