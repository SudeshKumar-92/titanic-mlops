#pip install pandas numpy matplotlib seaborn scikit-learn
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# --- Titanic dataset ---
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df_titanic = pd.read_csv(url)
print("=== Titanic ===")
print(df_titanic.head())


#print(df_titanic.isnull().sum() )  # count of nulls per column ) 

# --- Iris dataset ---
#data = load_iris()
#df_iris = pd.DataFrame(data.data, columns=data.feature_names)
#df_iris['target'] = data.target
#print("\n=== Iris ===")
#print(df_iris.head())

#Basic visualization:

#df_titanic['Age'].hist()               # distribution of Age
#plt.show()

#sns.heatmap(df_titanic.isnull())       # visual map of missing data
#plt.show()

#sns.pairplot(df_titanic)               # relationships between all columns
#plt.show()


#Cleaning data

# Drop rows with any missing values
df_titanic.dropna(inplace=True)

# Fill missing values with mean
df_titanic['Age'] = df_titanic['Age'].fillna(df_titanic['Age'].mean())

# Fill missing text with a placeholder
df_titanic['Name'] = df_titanic['Name'].fillna('Unknown')

# Remove duplicate rows
df_titanic.drop_duplicates(inplace=True)

# Rename a column
df_titanic.rename(columns={'Salary': 'salary'}, inplace=True)

# Change data type
df_titanic['Age'] = df_titanic['Age'].astype(int)

# Filter rows (only keep people over 18)
df_titanic = df_titanic[df_titanic['Age'] > 18]

# Drop a column you don't need
print(df_titanic.columns.tolist())

df_titanic = df_titanic.drop(columns=['Cabin'])

df_titanic = df_titanic.drop(columns=['unnecessary_column'], errors='ignore')


