# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Charger les données
data = pd.read_csv(r'C:\Users\elisa\OneDrive\Documents\P5\PS_20174392719_1491204439457_log.csv')

# Sélection des colonnes pertinentes (numériques) pour GMM
numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Sélection des features
X = data[numerical_cols]

# Standardisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Trouver le nombre optimal de composants en utilisant la méthode du coude (Elbow Method)
max_components = 10  # Vous pouvez ajuster cette valeur en fonction de votre cas d'utilisation
bic_values = []

for n_components in range(1, max_components + 1):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)
    bic_values.append(gmm.bic(X_scaled))

# Tracé du coude
import matplotlib.pyplot as plt

plt.plot(range(1, max_components + 1), bic_values, marker='o')
plt.xlabel('Nombre de composants')
plt.ylabel('BIC (Bayesian Information Criterion)')
plt.title('Méthode du coude pour le choix du nombre de composants (GMM)')
plt.show()

# Choisissez le nombre optimal de composants en fonction du coude dans le graphique
optimal_components = 3  # Mettez la valeur optimale que vous avez observée sur le graphique

# Appliquer GMM avec le nombre optimal de composants
gmm = GaussianMixture(n_components=optimal_components, random_state=42)
data['cluster'] = gmm.fit_predict(X_scaled)

# Analyse des résultats
silhouette_avg = silhouette_score(X_scaled, data['cluster'])
print(f'Silhouette Score: {silhouette_avg}')

fraud_cluster = data.groupby('cluster')['isFraud'].mean()
print(f'Taux de fraude par cluster:\n{fraud_cluster}')

# Vous pouvez analyser davantage les caractéristiques des clusters, etc.

# Si vous avez des étiquettes de fraude réelles, vous pouvez évaluer la performance
# en comparant les résultats de GMM avec les étiquettes réelles.
