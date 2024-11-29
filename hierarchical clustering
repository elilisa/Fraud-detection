# Import des bibliothèques nécessaires
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# Charger les données
data = pd.read_csv(r'C:\Users\elisa\OneDrive\Documents\P5\PS_20174392719_1491204439457_log.csv')
# Sélection des colonnes pertinentes (numériques) pour Hierarchical Clustering
numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Échantillonnage aléatoire
sample_size = 100000  # Ajustez la taille de l'échantillon en fonction de vos ressources disponibles
data_sample = data.sample(n=sample_size, random_state=42)

# Sélection des features
X = data_sample[numerical_cols]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Appliquer Hierarchical Clustering avec la méthode de liaison ward
linked = linkage(X_scaled, 'ward', metric='euclidean')

# Tracé du dendrogramme
import matplotlib.pyplot as plt

dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogramme pour Hierarchical Clustering')
plt.xlabel('Échantillons')
plt.ylabel('Distance euclidienne')
plt.show()

# Coupage du dendrogramme pour obtenir un nombre optimal de clusters
optimal_clusters = 3  # Vous pouvez ajuster cette valeur en fonction de votre cas d'utilisation
clusters = fcluster(linked, optimal_clusters, criterion='maxclust')

# Ajout des résultats au dataframe
data_sample['cluster'] = clusters

# Analyse des résultats
silhouette_avg = silhouette_score(X_scaled, data_sample['cluster'])
print(f'Silhouette Score: {silhouette_avg}')

fraud_cluster = data_sample.groupby('cluster')['isFraud'].mean()
print(f'Taux de fraude par cluster:\n{fraud_cluster}')

# Vous pouvez analyser davantage les caractéristiques des clusters, etc.

# Si vous avez des étiquettes de fraude réelles, vous pouvez évaluer la performance
# en comparant les résultats de Hierarchical Clustering avec les étiquettes réelles.
