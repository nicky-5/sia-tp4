from typing import Any, Callable

import math

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.kohonen import kohonen, RowDiff, ColDiff, Distance, Epoch

def euclidean(row_diff: RowDiff, col_diff: ColDiff) -> Distance:
    return np.sqrt(row_diff ** 2 + col_diff ** 2)

def constant(num: float) -> Callable[[Any], float]:
    return lambda _: num

def linear(start: float, end: float, width: int) -> Callable[[Epoch], float]:
    slope = (end - start) / width
    return lambda epoch: start + epoch * slope

def exponential_falloff(start: float, end: float, multiplier: float = 1) -> Callable[[Epoch], float]:
    scale = start - end
    return lambda epoch: start + scale * math.exp(- epoch * multiplier)

if __name__ == "__main__":
    df = pd.read_csv('./data/europe.csv')
    df_labels = df.loc[:, df.columns == 'Country']
    df_numeric = df.loc[:, df.columns != 'Country']

    df_numeric_normalized = (df_numeric - df_numeric.mean()) / df_numeric.std()
    df_normalized = pd.concat([df_labels, df_numeric_normalized], axis=1)

    inputs = df_normalized.shape[1]
    
    records_matrix, weights_matrix = kohonen(
        k=7,
        iterations=500 * inputs,
        df=df_normalized,
        distance=euclidean,
        radius=exponential_falloff(3, 1, 1/100),
        eta=constant(0.001),
        example=False,
        label="Country",
        rng=np.random.default_rng(843926515) # Pinned random seed to repeat results
    )

    # Assuming records_matrix is a 2D list
    data = np.empty((len(records_matrix), len(records_matrix[0])), dtype=object)

    for row, records_matrix_row in enumerate(records_matrix):
        for col, records in enumerate(records_matrix_row):
            if len(records) > 0:
                data[row, col] = '\n'.join([record[0] for record in records])  # use '\n' as the join

    # Create an empty 2D array to store the average distances
    avg_distances = np.zeros((len(weights_matrix), len(weights_matrix[0])))

    # Calculate the average distance for each neuron
    for i in range(len(weights_matrix)):
        for j in range(len(weights_matrix[0])):
            distances = []
            
            # Check all four neighbors (up, down, left, right)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                
                # If the neighbor is within the grid
                if 0 <= ni < len(weights_matrix) and 0 <= nj < len(weights_matrix[0]):
                    # Calculate the Euclidean distance
                    distance = np.linalg.norm(weights_matrix[i, j] - weights_matrix[ni, nj])
                    distances.append(distance)
            
            # Store the average distance
            avg_distances[i, j] = np.mean(distances)

    # Create the heatmap
    rows = 2
    cols = 4

    subplots = plt.subplots(rows, cols)
    axs: list[list[Axes]] = subplots[1]

    im = axs[rows-1][cols-1].imshow(avg_distances, cmap='hot')
    axs[rows-1][cols-1].set_title("Average distance")
    plt.colorbar(im)

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

                axs[rows-1][cols-1].text(j, i, data[i, j], ha="center", va="center", color=font_color)

    for index, col_name in enumerate(df_numeric.columns):
        row = int(index / cols)
        col = index % cols

        weight_matrix = weights_matrix[:,:,index]
        std = df_numeric.std()[col_name]
        mean = df_numeric.mean()[col_name]

        axs[row,col].set_title(col_name)
        sn.heatmap(weight_matrix * std + mean, ax=axs[row,col])

    plt.show()
