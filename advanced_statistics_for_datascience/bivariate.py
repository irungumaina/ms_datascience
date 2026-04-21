import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Task 1: Load the Iris dataset
iris = sns.load_dataset('iris')

# Focus on sepal length (X) and petal length (Y)
x = iris['sepal_length']
y = iris['petal_length']

# Task 2: Create a scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='species', palette='viridis')
plt.title('Scatter Plot: Sepal Length vs Petal Length')
plt.show()

# Task 3: Plot the joint distribution highlighting areas of high density
# Using Kernel Density Estimation (KDE) to visualize the continuous bivariate distribution
joint_plot = sns.jointplot(
    data=iris, 
    x='sepal_length', 
    y='petal_length', 
    hue='species', 
    kind='kde', 
    fill=True, 
    palette='viridis'
)
joint_plot.fig.suptitle('Joint Bivariate Density: Sepal vs. Petal Length', y=1.02)
plt.show()

# Calculate Pearson Correlation Coefficient
correlation, _ = pearsonr(x, y)
print(f"Pearson Correlation Coefficient: {correlation:.4f}")