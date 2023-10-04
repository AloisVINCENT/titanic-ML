import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # for file system operations
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#####################################################################
##################### Chargement des donnnées #######################
#####################################################################

# Load the data
train_data = pd.read_csv('input/train.csv')
# Load the test data
test_data = pd.read_csv('input/test.csv')

#####################################################################
################### Prétraitement des donnnées ######################
#####################################################################

# Vérifier le nombre d'élements vides de chaque colonne
#print(train_data.isnull().sum())
#print(test_data.isnull().sum())

#! Prendre appui sur : https://www.kaggle.com/code/hoanglongroai/79-accuracy-from-titanic-disaster/notebook

#TODO : Remplacer les valeurs manquantes de l'âge par des catégories en fonction des catégories sociales
# Remplacer les valeurs manquantes de l'âge par la médiane de l'âge
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

# Catégorisation de l'âge
def create_age_groups(df):
  age_group = list()
  for age in df["Age"]:
    if (age <= 18):
      age_group.append("child")
    elif (age <= 30):
      age_group.append("young")
    elif (age <= 45):
      age_group.append("adult")
    else:
      age_group.append("old")
  df["Age_group"] = age_group
  return df

train_data = create_age_groups(train_data)
test_data = create_age_groups(test_data)
print(train_data.head(5))

#TODO : Remplacer les valeurs manquantes de l'âge par des catégories en fonction de l'âge
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)

# Catégorisation du prix du billet
train_data['Fare'] = pd.cut(train_data['Fare'], bins=[0, 50, 100, 150, 300, 10000], labels=[0, 1, 2, 3, 4])
test_data['Fare'] = pd.cut(test_data['Fare'], bins=[0, 50, 100, 150, 300, 10000], labels=[0, 1, 2, 3, 4])
print(train_data['Fare'].value_counts())


# Encodage one-hot du port d'embarcation
X = pd.get_dummies(train_data["Embarked"], columns=["Embarked"], drop_first=True)
X_test = pd.get_dummies(test_data["Embarked"], columns=["Embarked"], drop_first=True)

features = ["Pclass", "Sex", "SibSp", "Parch", "Age", "Embarked"]



# Remplacer les valeurs manquantes du port d'embarcation par la valeur la plus fréquente
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])


#####################################################################
################## Création du modèle de forêt ######################
#####################################################################


#Create the model and set the parameters
forest_model = RandomForestClassifier(n_estimators= 5000, max_depth = 6, random_state=1)

#Train the model

y = train_data["Survived"]

print("\nPredicting using RandomForest...")
forest_model.fit(X, y)
forest_pred = forest_model.predict(X_test)
print("Done predicting using RandomForest.\n")

# #####################################################################
# ################## Création du modèle de GBM ########################
# #####################################################################

# # Créer le GBM
# gbm_model = GradientBoostingClassifier(n_estimators=500, max_depth=5, random_state=1)

# # Entraîner le GBM
# print("Predicting using GBM...")
# gbm_model.fit(X, y)

# # Faire des prédictions
# gbm_pred = gbm_model.predict(X_test)
# print("Done predicting using GBM.\n")

# #####################################################################
# ########### Création du modèle de Régression logistique #############
# #####################################################################


# # Créer le modèle de régression logistique
# logistic_model = LogisticRegression(max_iter=1000)


# # Entraîner le modèle sur l'ensemble d'entraînement
# print("Predicting using LogisticRegression...")
# logistic_model.fit(X, y)

# # Faire des prédictions sur l'ensemble de test
# logistic_pred = logistic_model.predict(X_test)
# print("Done predicting using LogisticRegression.\n")

# #####################################################################
# ###################### Cross-Validation #############################
# #####################################################################
# """
# k = 5
# #Calculer le score de chaque modèle
# print("Calculating RandomForest score...")
# forest_model_score = cross_val_score(forest_model, X, y, cv=k, scoring = 'accuracy')
# print("Calculating GBM score...")
# gbm_model_score = cross_val_score(gbm_model, X, y, cv=k, scoring = 'accuracy')
# print("Calculating LogisticRegression score...\n")
# logistic_model_score = cross_val_score(logistic_model, X, y, cv=k, scoring = 'accuracy')

# #Calculer la moyenne des scores
# mean_forest_model_score = np.mean(forest_model_score)
# mean_gbm_model_score = np.mean(gbm_model_score)
# mean_logistic_model_score = np.mean(logistic_model_score)

# print("RandomForest mean Accuracy:", mean_forest_model_score)
# print("GBM mean Accuracy:", mean_gbm_model_score)
# print("LogisticRegression mean Accuracy:", mean_logistic_model_score)
# """
# #####################################################################
# ##################### Création du Stacking ##########################
# #####################################################################

# # Créer un nouveau jeu de données avec les prédictions des 3 modèles
# X_train_stack = X.copy()
# X_train_stack['RandomForest'] = forest_model.predict(X)
# X_train_stack['GBM'] = gbm_model.predict(X)
# X_train_stack['LogisticRegression'] = logistic_model.predict(X)

# X_test_stack = X_test.copy()
# X_test_stack['RandomForest'] = forest_model.predict(X_test)
# X_test_stack['GBM'] = gbm_model.predict(X_test)
# X_test_stack['LogisticRegression'] = logistic_model.predict(X_test)

# print("Predicting using level 2 Stacking...")
# level2_model = RandomForestClassifier(n_estimators= 500, max_depth = 5, random_state=1)
# level2_model.fit(X_train_stack, y)

# lvl2_pred = level2_model.predict(X_test_stack)
# print("Done predicting using level 2 Stacking.\n")

#####################################################################
##################### Création de l'output ##########################
#####################################################################

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': forest_pred})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
