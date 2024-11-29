# Import des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Charger les données
data = pd.read_csv(r'C:\Users\elisa\OneDrive\Documents\P5\PS_20174392719_1491204439457_log.csv')

# Encodage des colonnes 'nameOrig' et 'nameDest'
label_encoder = LabelEncoder()
data['nameOrig'] = label_encoder.fit_transform(data['nameOrig'])
data['nameDest'] = label_encoder.fit_transform(data['nameDest'])

# Conversion des variables catégorielles en variables indicatrices (one-hot encoding)
data = pd.get_dummies(data, columns=['type'])

# Séparation des features et de la variable cible
X = data.drop(['isFraud'], axis=1)
y = data['isFraud']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation du modèle LightGBM
model = LGBMRegressor()

# Entraînement du modèle
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Arrondir les prédictions à 0 ou 1, car il s'agit d'une tâche de classification binaire
y_pred_binary = [1 if val >= 0.5 else 0 for val in y_pred]

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred_binary)
conf_matrix = confusion_matrix(y_test, y_pred_binary)
classification_rep = classification_report(y_test, y_pred_binary)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
