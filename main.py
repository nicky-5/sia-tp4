from typing import Any, Callable

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from src.kohonen import kohonen, RowDiff, ColDiff, Distance

def euclidean(row_diff: RowDiff, col_diff: ColDiff) -> Distance:
    return np.sqrt(row_diff ** 2 + col_diff ** 2)

def constant(num: float) -> Callable[[Any], float]:
    return lambda _: num

if __name__ == "__main__":
    df = pd.read_csv('./data/europe.csv')
    df_labels = df.loc[:, df.columns == 'Country']
    df_numeric = df.loc[:, df.columns != 'Country']

    df_numeric_normalized = (df_numeric - df_numeric.mean()) / df_numeric.std()
    df_normalized = pd.concat([df_labels, df_numeric_normalized], axis=1)

    inputs = df_normalized.shape[1]
    
    records_matrix, weights_matrix = kohonen(
        k=4,
        iterations=10000 * inputs,
        df=df_normalized,
        distance=euclidean,
        radius=2,
        eta=0.1,
        example=False,
        label="Country"
    )
    # Assuming records_matrix is a 2D list
    data = np.empty((len(records_matrix), len(records_matrix[0])), dtype=object)

    for row, records_matrix_row in enumerate(records_matrix):
        for col, records in enumerate(records_matrix_row):
            if len(records) > 0:
                data[row, col] = '\n'.join([record[0] for record in records])  # use '\n' as the join

    fig, ax = plt.subplots()
    im = ax.imshow(np.random.random(data.shape), cmap='hot')  # random data for colors

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(records_matrix[0])))
    ax.set_yticks(np.arange(len(records_matrix)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(range(len(records_matrix[0])))
    ax.set_yticklabels(range(len(records_matrix)))

    # Loop over data dimensions and create text annotations.
    for i in range(len(records_matrix)):
        for j in range(len(records_matrix[0])):
            if data[i, j] is not None:
                # Get the color of the current cell
                cell_color = im.cmap(im.norm(im.get_array()[i, j]))

                # Calculate the brightness of the color
                brightness = (cell_color[0]*299 + cell_color[1]*587 + cell_color[2]*114) / 1000

                # Use white font if the cell is dark, otherwise use black
                font_color = "w" if brightness < 0.5 else "k"

                text = ax.text(j, i, data[i, j],
                            ha="center", va="center", color=font_color)

    plt.show()


    # Assuming som is your Kohonen network and it has a method get_weights() that returns a 2D array of weight vectors
    weights = weights_matrix

    # Create an empty 2D array to store the average distances
    avg_distances = np.zeros((len(weights), len(weights[0])))

    # Calculate the average distance for each neuron
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            distances = []
            
            # Check all four neighbors (up, down, left, right)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                
                # If the neighbor is within the grid
                if 0 <= ni < len(weights) and 0 <= nj < len(weights[0]):
                    # Calculate the Euclidean distance
                    distance = np.linalg.norm(weights[i, j] - weights[ni, nj])
                    distances.append(distance)
            
            # Store the average distance
            avg_distances[i, j] = np.mean(distances)

    # Create the heatmap
    plt.imshow(avg_distances, cmap='hot')
    plt.colorbar(label='Average distance')
    plt.show()
    rows = 2
    cols = 4
    _, axs = plt.subplots(rows, cols)

    for index, col_name in enumerate(df_numeric.columns):
        row = int(index / cols)
        col = index % cols

        weight_matrix = weights_matrix[:,:,index]
        std = df_numeric.std()[col_name]
        mean = df_numeric.mean()[col_name]

        axs[row,col].set_title(col_name)
        sn.heatmap(weight_matrix * std + mean, ax=axs[row,col])

    heatmap = np.vectorize(len)(records_matrix)
    axs[rows-1,cols-1].set_title("Heatmap")
    sn.heatmap(heatmap, ax=axs[rows-1,cols-1], annot=True)

    plt.show()
