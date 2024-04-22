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
