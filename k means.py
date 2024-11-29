# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Charger les données
data = pd.read_csv(r'C:\Users\elisa\OneDrive\Documents\P5\PS_20174392719_1491204439457_log.csv')

# Sélection des colonnes pertinentes (numériques) pour K-means
numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Sélection des features
X = data[numerical_cols]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Trouver le nombre optimal de clusters en utilisant la méthode du coude (Elbow Method)
max_clusters = 10  # Vous pouvez ajuster cette valeur en fonction de votre cas d'utilisation
inertia_values = []

for n_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

# Tracé du coude
import matplotlib.pyplot as plt

plt.plot(range(1, max_clusters + 1), inertia_values, marker='o')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour le choix du nombre de clusters')
plt.show()

# Choisissez le nombre optimal de clusters en fonction du coude dans le graphique
optimal_clusters = 3  # Mettez la valeur optimale que vous avez observée sur le graphique

# Appliquer K-means avec le nombre optimal de clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

# Analyse des résultats
silhouette_avg = silhouette_score(X_scaled, data['cluster'])
print(f'Silhouette Score: {silhouette_avg}')

fraud_cluster = data.groupby('cluster')['isFraud'].mean()
print(f'Taux de fraude par cluster:\n{fraud_cluster}')

# Vous pouvez analyser davantage les caractéristiques des clusters, etc.

# Si vous avez des étiquettes de fraude réelles, vous pouvez évaluer la performance
# en comparant les résultats de K-means avec les étiquettes réelles.
