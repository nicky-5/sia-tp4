from src.linear_perceptron import LinearPerceptron
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('./data/europe.csv')

    # Separate features (excluding 'Country')
    X = df.drop(columns=['Country'])

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    eigenvectors = pca.components_

    # Esto es para comparar los resultados obtenidos por nuestro perceptron
    # No se puede usar la libreria para resolver el problema
    print("Eigenvectors (principal components):")
    print(eigenvectors)





    # Assuming 'Country' is the column to exclude
    df_numeric = df.loc[:, df.columns != 'Country']

    scaler = StandardScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df_numeric), columns=df_numeric.columns)

    print(df_normalized)

    linear_perceptron = LinearPerceptron(
        df_normalized.to_numpy(),
        [],
        learning_rate=0.1
    )
    epochs, converged = linear_perceptron.train(1000)

    print("\n----- LinearPerceptron -----\n")

    if not converged:
        print(f"Did not converge after {epochs} epochs\n")
    else:
        print(f"Finished learning at {epochs} epochs\n")
        print("Output: ", linear_perceptron.get_outputs())
        print("Weights: ", linear_perceptron.weights)

    # print(linear_perceptron)

    w = linear_perceptron.weights

    print(w / np.sqrt(np.inner(w, w)))
    print(eigenvectors[0])
