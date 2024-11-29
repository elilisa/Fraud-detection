import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Charger les données
data = pd.read_csv(r'C:\Users\elisa\OneDrive\Documents\P5\PS_20174392719_1491204439457_log.csv')

# Sélection des colonnes pertinentes pour Apriori
transaction_data = data[['type', 'amount']]

# Transformation des données catégoriques en données transactionnelles
transactions = transaction_data.groupby(['type', 'amount']).size().unstack().reset_index().fillna(0).set_index('type')

# Convertir les comptages en binaire (1 si l'élément est présent dans la transaction, 0 sinon)
transactions = transactions.applymap(lambda x: 1 if x > 0 else 0)

# Appliquer l'algorithme Apriori
frequent_itemsets = apriori(transactions, min_support=0.01, use_colnames=True)

# Générer des règles d'association
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Afficher les règles d'association
print(rules)
