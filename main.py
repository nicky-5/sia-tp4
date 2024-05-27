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
        iterations=500 * inputs,
        df=df_normalized,
        distance=euclidean,
        radius=constant(2),
        eta=constant(0.005),
        example=False,
        label="Country"
    )

    for row, records_matrix_row in enumerate(records_matrix):
        for col, records in enumerate(records_matrix_row):
            if len(records) > 0:
                print("Countries in row {} column {}".format(row, col))
                print([record[0] for record in records])
                print()

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
