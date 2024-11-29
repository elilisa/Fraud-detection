import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score   

data = pd.read_csv(r'C:\Users\elisa\OneDrive\Documents\P5\PS_20174392719_1491204439457_log.csv') #import du dtatset

# Supprimer les colonnes non nécessaires (si nécessaire)
data = data.drop(['nameOrig', 'nameDest'], axis=1)

# Convertir les variables catégorielles en variables indicatrices (si nécessaire)
data = pd.get_dummies(data, columns=['type'], drop_first=True)

# Diviser les données en features (X) et la variable cible (y)
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

# Initialiser le modèle Random Forest
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)

cv_scores = cross_val_score(random_forest_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

# Entraîner le modèle
random_forest_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
predictions = random_forest_model.predict(X_test)

# Évaluer la performance du modèle
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
