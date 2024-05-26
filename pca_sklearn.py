import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm

# Load the dataset from the uploaded CSV file
file_path = './data/europe.csv'
df = pd.read_csv(file_path)

# Extract country names and features
countries = df['Country']
X = df.drop(columns=['Country'])

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Function to create a biplot with unique colors for each country
def biplot(X_pca, pca, countries, show_names=True):
    plt.figure(figsize=(10, 8), dpi= 80)
    
    # Generate a unique color for each country
    colors = cm.rainbow(np.linspace(0, 1, len(countries)))
    
    # Plot the principal components
    for i, country in enumerate(countries):
        plt.scatter(X_pca[i, 0], X_pca[i, 1], color=colors[i], edgecolor='k', s=50)
        if show_names:
            plt.text(X_pca[i, 0] + 0.02, X_pca[i, 1] + 0.02, country, fontsize=9)
    
    # Plot the loadings (arrows)
    feature_vectors = pca.components_.T
    for i, v in enumerate(feature_vectors):
        plt.arrow(0, 0, v[0], v[1], color='red', alpha=0.5, head_width=0.05)
        plt.text(v[0] + 0.02, v[1] + 0.02, X.columns[i], color='red', fontsize=12)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Biplot')
    plt.grid()
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.show()

# Create a biplot with country names shown and unique colors
biplot(X_pca, pca, countries, show_names=True)

# Create a biplot with country names hidden and unique colors
biplot(X_pca, pca, countries, show_names=False)

# Create a barplot of the first principal component for each country
plt.figure(figsize=(12, 8), dpi=80)
bars = plt.bar(countries, X_pca[:, 0], color=cm.rainbow(np.linspace(0, 1, len(countries))), edgecolor='k')
plt.xlabel('Country')
plt.ylabel('Principal Component 1')
plt.title('Principal Component 1 of Each Country')
plt.xticks(rotation=90)
plt.grid(axis='y')

# Add text labels on the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), ha='center', va='bottom')

plt.show()
