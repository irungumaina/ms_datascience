import pandas as pd

# Load the dataset
df = pd.read_csv('fiia.csv')

# Inspect the first few rows and column info
print(df.head())
print(df.info())
print(df['country'].unique())