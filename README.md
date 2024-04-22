Pandas est une bibliothèque Python populaire utilisée pour la manipulation et l'analyse des données. Voici un aperçu des bases de Pandas pour vous aider à démarrer :

### 1. Les structures de données principales :

Pandas offre deux principales structures de données :

- **Series** : C'est un tableau unidimensionnel qui peut contenir des données de tout type (entiers, chaînes de caractères, valeurs flottantes, etc.). Une Series est composée de données et d'indices (étiquettes).

- **DataFrame** : C'est une structure de données bidimensionnelle similaire à une table de base de données ou à une feuille de calcul Excel. Elle est constituée de lignes et de colonnes, et chaque colonne peut avoir un type de données différent.

### 2. Importation de Pandas :

Avant de pouvoir utiliser Pandas, vous devez l'importer dans votre script Python ou dans votre environnement de travail. La convention courante est d'importer Pandas sous le nom `pd` :

```python
import pandas as pd
```

### 3. Création de Series et de DataFrame :

Vous pouvez créer une Series en passant une liste de valeurs à la fonction `pd.Series()` :

```python
s = pd.Series([1, 3, 5, 7, 9])
```

Pour créer un DataFrame, vous pouvez passer un dictionnaire où les clés sont les noms des colonnes et les valeurs sont les données, à la fonction `pd.DataFrame()` :

```python
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}
df = pd.DataFrame(data)
```

### 4. Chargement de données :

Pandas offre des fonctions pour charger des données à partir de différents types de fichiers, tels que CSV, Excel, JSON, SQL, etc. Par exemple, pour charger un fichier CSV :

```python
df = pd.read_csv('data.csv')
```

### 5. Exploration des données :

Vous pouvez utiliser diverses méthodes pour explorer et comprendre vos données, telles que `head()`, `tail()`, `info()`, `describe()`, `shape`, etc. Par exemple :

```python
print(df.head())  # Affiche les premières lignes du DataFrame
print(df.info())  # Affiche des informations sur les colonnes et les types de données
print(df.describe())  # Affiche des statistiques descriptives pour les données numériques
```

### 6. Manipulation des données :

Pandas offre de nombreuses fonctionnalités pour manipuler et transformer les données, telles que la sélection, le filtrage, le tri, l'agrégation, le nettoyage des données manquantes, la fusion, la concaténation, etc.

### 7. Visualisation des données :

Bien que Pandas soit principalement axé sur la manipulation de données, il offre également des fonctionnalités de visualisation intégrées grâce à la prise en charge de Matplotlib.

# methode de statistique
 Pandas offre une variété de méthodes d'analyse pour explorer et comprendre les données. Voici quelques-unes des méthodes d'analyse les plus couramment utilisées :

### 1. `describe()` :
Cette méthode calcule des statistiques descriptives pour les données numériques telles que la moyenne, l'écart type, les quantiles, etc.

Exemple :
```python
print(df.describe())
```

### 2. `value_counts()` :
Cette méthode compte le nombre d'occurrences de chaque valeur unique dans une Series.

Exemple :
```python
print(df['column_name'].value_counts())
```

### 3. `groupby()` :
Cette méthode permet de regrouper les données en fonction des valeurs d'une ou plusieurs colonnes, puis d'appliquer des fonctions d'agrégation sur chaque groupe.

Exemple :
```python
grouped = df.groupby('column_name')
print(grouped.mean())
```

### 4. `pivot_table()` :
Cette méthode crée une table pivot à partir d'un DataFrame, ce qui permet de réorganiser et d'agréger les données en fonction des valeurs de certaines colonnes.

Exemple :
```python
pivot_table = df.pivot_table(index='column1', columns='column2', values='value', aggfunc='mean')
print(pivot_table)
```

### 5. `corr()` :
Cette méthode calcule la corrélation entre les colonnes numériques d'un DataFrame.

Exemple :
```python
print(df.corr())
```

### 6. `isnull()` et `notnull()` :
Ces méthodes permettent de détecter les valeurs nulles ou non nulles dans un DataFrame.

Exemple :
```python
print(df.isnull().sum())
```

### 7. `rolling()` :
Cette méthode permet de calculer des statistiques mobiles telles que la moyenne mobile, l'écart-type mobile, etc.

Exemple :
```python
rolling_mean = df['column'].rolling(window=3).mean()
print(rolling_mean)
```

### 8. `diff()` :
Cette méthode calcule la différence entre les éléments consécutifs d'une Series ou d'un DataFrame.

Exemple :
```python
diff = df['column'].diff()
print(diff)
```

### 9. `cumsum()` et `cumprod()` :
Ces méthodes calculent respectivement la somme cumulée et le produit cumulé des éléments d'une Series ou d'un DataFrame.

Exemple :
```python
cumulative_sum = df['column'].cumsum()
print(cumulative_sum)
```

Ces méthodes constituent une partie importante de l'arsenal d'analyse de données offert par Pandas. Elles vous permettent d'explorer vos données, d'identifier des tendances, des corrélations et des modèles, et de prendre des décisions éclairées en fonction de ces analyses.

# Graphe
Pour créer des graphiques à partir des résultats que vous avez trouvés en utilisant Pandas, vous pouvez utiliser la bibliothèque de visualisation Matplotlib intégrée à Pandas, ou d'autres bibliothèques de visualisation comme Seaborn ou Plotly. Voici quelques exemples de graphiques que vous pouvez créer avec Pandas et Matplotlib :

### Histogramme :
Un histogramme est un graphique qui montre la répartition des valeurs dans une série de données.

```python
import matplotlib.pyplot as plt

# Créer un histogramme
df['column'].hist()

# Ajouter des labels et un titre
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Histogram')

# Afficher le graphique
plt.show()
```

### Diagramme à barres :
Un diagramme à barres est utile pour comparer les valeurs entre différentes catégories.

```python
# Créer un diagramme à barres
df['column'].value_counts().plot(kind='bar')

# Ajouter des labels et un titre
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Bar Chart')

# Afficher le graphique
plt.show()
```

### Diagramme circulaire :
Un diagramme circulaire montre la répartition des valeurs en pourcentage.

```python
# Créer un diagramme circulaire
df['column'].value_counts().plot(kind='pie')

# Ajouter un titre
plt.title('Pie Chart')

# Afficher le graphique
plt.show()
```

### Diagramme de dispersion :
Un diagramme de dispersion est utilisé pour visualiser la relation entre deux variables.

```python
# Créer un diagramme de dispersion
plt.scatter(df['column1'], df['column2'])

# Ajouter des labels et un titre
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Scatter Plot')

# Afficher le graphique
plt.show()
```

### Ligne ou courbe :
Un graphique en ligne est utilisé pour afficher l'évolution d'une variable sur une période.

```python
# Créer un graphique en ligne
df['column'].plot(kind='line')

# Ajouter des labels et un titre
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Line Plot')

# Afficher le graphique
plt.show()
```

Ces exemples de graphiques utilisent Matplotlib pour la visualisation, mais vous pouvez également utiliser d'autres bibliothèques de visualisation comme Seaborn ou Plotly pour créer des graphiques plus complexes et esthétiques. En utilisant ces méthodes, vous pouvez explorer visuellement vos données et en extraire des insights importants.


# EXEMPLE D'APPLICATION
Avec le jeu de données Titanic, vous pouvez créer un modèle de prédiction de survie en fonction de caractéristiques telles que l'âge, le sexe et la classe. Voici comment vous pouvez le faire en utilisant Python et Scikit-learn :

### 1. Chargement des données et préparation :

```python
import pandas as pd

# Charger les données Titanic depuis un fichier CSV
titanic_data = pd.read_csv('titanic.csv')

# Sélectionner les caractéristiques pertinentes (âge, sexe, classe) et la cible (survie)
data = titanic_data[['Age', 'Sex', 'Pclass', 'Survived']].copy()

# Convertir le sexe en valeurs numériques (0 pour homme, 1 pour femme)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()

# Séparer les caractéristiques et la cible
X = data[['Age', 'Sex', 'Pclass']]
y = data['Survived']
```

### 2. Division des données en ensembles d'entraînement et de test :

```python
from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 3. Entraînement du modèle de prédiction :

```python
from sklearn.ensemble import RandomForestClassifier

# Créer et entraîner un classificateur RandomForest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 4. Évaluation du modèle :

```python
from sklearn.metrics import accuracy_score

# Faire des prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Calculer l'exactitude du modèle
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

### 5. Utilisation du modèle pour faire des prédictions :

```python
# Exemple de prédiction pour un passager : [Age, Sexe, Classe]
prediction = model.predict([[30, 0, 3]])
print("Prediction:", prediction)
```

Avec ces étapes, vous avez créé un modèle de prédiction de survie basé sur l'âge, le sexe et la classe des passagers du Titanic. Vous pouvez ajuster les hyperparamètres du modèle, explorer d'autres algorithmes d'apprentissage automatique et ajouter des fonctionnalités supplémentaires pour améliorer les performances du modèle.
