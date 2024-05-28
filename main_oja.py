from src.linear_perceptron import LinearPerceptron
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.perceptron import constant_learning_rate, iterative_learning_rate

plt.rcParams['figure.dpi'] = 250


def manual_standard_scaler(df):
    means = df.mean()
    stds = df.std(ddof=0)
    df_normalized = (df - means) / stds
    return df_normalized

# Define a function to calculate RMSE


def rmse(predicted, expected):
    return np.sqrt(np.mean((predicted - expected) ** 2))


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('./data/europe.csv')

    # Separate features (excluding 'Country')
    X = df.drop(columns=['Country'])
    countries = df["Country"]

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
    df_normalized_sklearn = pd.DataFrame(
        scaler.fit_transform(df_numeric), columns=df_numeric.columns)
    df_normalized = pd.DataFrame(
        manual_standard_scaler(df_numeric), columns=df_numeric.columns)

    print(df_normalized)
    df_to_test = df_normalized

    def calc_pc1(df, lr, epochs, lr_fun):
        linear_perceptron = LinearPerceptron(
            df.to_numpy(),
            [],
            learning_rate=lr,
            learning_rate_fun=lr_fun
        )
        linear_perceptron.train(epochs)

        # print(linear_perceptron)

        w = linear_perceptron.weights
        pc1 = w / np.sqrt(np.inner(w, w))
        if (pc1[0] < 0):
            pc1 = pc1 * -1
        return pc1

    def iter_lr():
        lr_list = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6]
        lr_fun = iterative_learning_rate
        epochs = 1000
        iterations = 10

        error_results = {}

        for lr in lr_list:
            error_results[lr] = {'iterative': [], 'constant': []}
            for _ in range(iterations):
                print(lr)
                lr_fun = iterative_learning_rate
                pc1 = calc_pc1(df_normalized, lr, epochs, lr_fun)
                error = rmse(pc1, eigenvectors[0])
                error_results[lr]['iterative'].append(error)

                lr_fun = constant_learning_rate
                pc1 = calc_pc1(df_normalized, lr, epochs, lr_fun)
                error = rmse(pc1, eigenvectors[0])
                error_results[lr]['constant'].append(error)

        # Calculate mean error for each configuration
        mean_error_results = {}
        for lr, errors in error_results.items():
            mean_error_results[lr] = {
                'iterative': np.mean(errors['iterative']),
                'constant': np.mean(errors['constant'])
            }
        std_error_results = {}
        for lr, errors in error_results.items():
            std_error_results[lr] = {
                'iterative': np.std(errors['iterative']),
                'constant': np.std(errors['constant'])
            }

        print("Mean RMSE for different learning rates and learning rate strategies:")
        print(mean_error_results)
        return mean_error_results, std_error_results

    def iter_epochs():
        lr_list = [10**-1, 10**-2, 10**-3, 10**-4, 10**-5, 10**-6]
        lr_fun = iterative_learning_rate
        epochs_list = [1000, 10000, 100000]
        iterations = 3

        error_results = {}

        for epochs in epochs_list:
            error_results[epochs] = {'iterative': [], 'constant': []}
            for _ in range(iterations):
                print(epochs)
                lr = 0.1
                lr_fun = iterative_learning_rate
                pc1 = calc_pc1(df_normalized, lr, epochs, lr_fun)
                error = rmse(pc1, eigenvectors[0])
                error_results[epochs]['iterative'].append(error)

                lr = 10**-3
                lr_fun = constant_learning_rate
                pc1 = calc_pc1(df_normalized, lr, epochs, lr_fun)
                error = rmse(pc1, eigenvectors[0])
                error_results[epochs]['constant'].append(error)

        # Calculate mean error for each configuration
        mean_error_results = {}
        for epochs, errors in error_results.items():
            mean_error_results[epochs] = {
                'iterative': np.mean(errors['iterative']),
                'constant': np.mean(errors['constant'])
            }
        std_error_results = {}
        for epochs, errors in error_results.items():
            std_error_results[epochs] = {
                'iterative': np.std(errors['iterative']),
                'constant': np.std(errors['constant'])
            }

        print("Mean RMSE for different epochs and learning rate strategies:")
        print(mean_error_results)
        return mean_error_results, std_error_results

    mean_error_results = None
    std_error_results = None
    # mean_error_results, std_error_results = iter_lr()
    if (mean_error_results is not None):
        # Extracting data for plotting
        lr_values = list(mean_error_results.keys())
        iterative_errors = [mean_error_results[lr]['iterative']
                            for lr in lr_values]
        constant_errors = [mean_error_results[lr]['constant'] for lr in lr_values]

        # Line plot
        # Extracting standard deviation for error bars
        iterative_errors_std = [std_error_results[lr]
                                ['iterative'] for lr in lr_values]
        constant_errors_std = [std_error_results[lr]['constant']
                               for lr in lr_values]

        plt.figure(figsize=(10, 6))
        plt.errorbar(lr_values, iterative_errors, yerr=iterative_errors_std,
                     label='Iterative', color='blue', marker='o', capsize=5)
        plt.errorbar(lr_values, constant_errors, yerr=constant_errors_std,
                     label='Constant', color='red', marker='s', capsize=5)

        # Title and labels
        plt.title('Mean RMSE for Different Learning Rate and Strategies')
        plt.xlabel('Learning Rate (Log Scale)')
        plt.ylabel('Mean RMSE')
        plt.xscale('log')  # Set logarithmic scale for x-axis
        plt.grid(True, which='both', linestyle='--',
                 linewidth=0.5)  # Add gridlines
        plt.legend()  # Show legend
        plt.tight_layout()  # Adjust layout to prevent clipping

        # Show plot
        plt.show()
        exit()

    lr = 0.1
    epochs = 10000
    lr_fun = iterative_learning_rate

    pc1 = calc_pc1(df_to_test, lr, epochs, lr_fun)

    print(pc1)
    print(eigenvectors[0])

    error = rmse(pc1, eigenvectors[0])
    print("rmse:", error)

    def row_dot_product(row, vector):
        return np.dot(row, vector)

    # Apply the dot product function to each row
    vals = df_normalized.apply(lambda row: row_dot_product(row, pc1), axis=1)
    print(vals)

    # Create the bar plot
    plt.figure(figsize=(12, 8), dpi=150)
    # You can customize the color
    bars = plt.bar(countries, vals, color=cm.rainbow(
        np.linspace(0, 1, len(countries))), edgecolor='k')

    # Add title and labels
    plt.title('PC1 by Country')
    plt.xlabel('Country')
    plt.ylabel('PC1')

    plt.xticks(rotation=90)
    plt.grid(axis='y')

    # Add text labels on the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval,
                 round(yval, 2), ha='center', va='bottom')

    # Show the plot
    plt.show()
