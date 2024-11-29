# **Détection de Fraude Financière avec le Machine Learning**

Ce projet explore l'application du machine learning pour identifier et prévenir la fraude financière. En exploitant des données volumineuses et des modèles algorithmiques avancés, nous développons des stratégies de détection précoce pour renforcer la sécurité dans le secteur financier moderne.

---

## **Objectifs du Projet**
1. Détecter les schémas frauduleux dans les transactions financières.
2. Comparer différentes approches algorithmiques pour identifier la méthode la plus performante.
3. Offrir un aperçu des techniques de prétraitement des données pour optimiser les modèles.
4. Réduire les faux positifs pour éviter les perturbations dans les opérations légitimes.

---

## **Structure du Projet**
Le projet est organisé en plusieurs étapes clés :
### **I. Récupération des Données**
- **Sources** : Les datasets utilisés sont récupérés sur [Kaggle](https://www.kaggle.com/).
- **Datasets utilisés** :
  - **Synthétique** : Pour explorer les comportements typiques des modèles.
  - **Large** : Simule un volume élevé de transactions bancaires réelles.
  - **Avec données manquantes** : Évaluer la robustesse des modèles en cas d'imperfections dans les données.

### **II. Nettoyage des Données**
Les datasets bruts sont prétraités pour améliorer la qualité des entrées dans les modèles. 
Trois approches de traitement des données manquantes ont été testées :
1. **Suppression** des lignes contenant des données manquantes.
2. **Imputation par la moyenne** des valeurs existantes.
3. **Imputation itérative sophistiquée** (préférée pour sa précision accrue).

### **III. Réduction de Dimension**
Réduire le nombre de caractéristiques pour :
- **Accroître la performance** : Moins de données à traiter.
- **Éliminer le bruit** : Se concentrer sur les variables pertinentes.
- **Éviter la malédiction de la dimension** : Réduction des risques de sur-adaptation (overfitting).

### **IV. Modèles de Machine Learning**
Treize algorithmes différents ont été implémentés et testés pour prédire les fraudes financières :
- **Clustering** :
  - `GMM.py` : Modèle de mélange gaussien.
  - `k means.py` : Algorithme des k-means.
  - `hierarchical clustering.py` : Clustering hiérarchique.
- **Régression** :
  - `linear regression.py` : Régression linéaire.
  - `ridge regression.py` : Régression Ridge.
  - `lasso regression.py` : Régression Lasso.
  - `gradient boosting regression.py` : Régression avec Gradient Boosting.
  - `light GBM regressor.py` : Régression avec LightGBM.
- **Classification** :
  - `logistic regression.py` : Régression logistique.
  - `decision tree.py` : Arbre de décision.
  - `random forest.py` : Forêt aléatoire.
  - `xgboost.py` : Algorithme XGBoost.
- **Association** :
  - `apriori algo.py` : Algorithme Apriori pour l'analyse des règles d'association.

---

## **Installation**
### **1. Pré-requis**
- Python 3.8 ou supérieur.
- Bibliothèques nécessaires (installer avec pip) :
  ```bash
  pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm
