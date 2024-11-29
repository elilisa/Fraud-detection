import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from fancyimpute import IterativeImputer


data = pd.read_csv(r'C:\Users\elisa\OneDrive\Documents\P5\PS_20174392719_1491204439457_log.csv') #import du dtatset


data_deleted = data.dropna() #on supprime les lignes à valeurs manquantes

numeric_features = data.select_dtypes(include=['float64']).columns
categorical_features = data.select_dtypes(include=['object']).columns #on separe les colonnes numeriques et non nuleriques


numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())                    #transformer colonnes numeriques
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())                             #transformer les colonnes non numeriques
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

data_mean_imputed = preprocessor.fit_transform(data) #appliquer les transfo

imputer = IterativeImputer(max_iter=10, random_state=0)
data_sophisticated_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)  # Technique sophistiquée pour gérer les valeurs manquantes

print("Statistiques descriptives après suppression des lignes avec valeurs manquantes:\n", data_deleted.describe())
print("\nStatistiques descriptives après imputation avec moyenne:\n", pd.DataFrame(data_mean_imputed, columns=numeric_features.tolist() + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_features))).describe())
print("\nStatistiques descriptives après technique sophistiquée pour trouver la donnée manquante:\n", data_sophisticated_imputed.describe())

data_deleted.to_csv('dataset_deleted.csv', index=False)
pd.DataFrame(data_mean_imputed, columns=numeric_features.tolist() + list(preprocessor.named_transformers_['cat']['onehot'].get_feature_names(categorical_features))).to_csv('dataset_mean_imputed.csv', index=False)
data_sophisticated_imputed.to_csv('dataset_sophisticated_imputed.csv', index=False)  #sauvegarder le nv dataset
