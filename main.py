from typing import Any

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from src.kohonen import *

def euclidean(row_diff: RowDiff, col_diff: ColDiff) -> Distance:
    return np.sqrt(row_diff ** 2 + col_diff ** 2)

def constant(num: float) -> Callable[[Any], float]:
    return lambda _: num

if __name__ == "__main__":
    df = pd.read_csv('./data/europe.csv')
    df_numeric = df.loc[:, df.columns != 'Country']
    df_normalized = (df_numeric - df_numeric.mean()) / df_numeric.std()

    inputs = df_normalized.shape[1]

    print(df_normalized)
    
    result_matrix = kohonen(
        k=5,
        iterations=500 * inputs,
        df=df_normalized,
        distance=euclidean,
        radius=constant(2),
        eta=constant(0.7),
        example=True
    )

    # heatmap = np.vectorize(len)(result_matrix)
    # sn.heatmap(heatmap)
    # plt.show()

    columns = 4
    _, axs = plt.subplots(2, columns)

    for index, title in enumerate(df_normalized.columns):
        row = int(index / columns)
        col = index % columns

        average = lambda records: np.average([rec[index] for rec in records])
        avg_matrix = np.vectorize(average)(result_matrix)

        axs[row,col].set_title(title)
        sn.heatmap(avg_matrix, ax=axs[row,col])

    plt.show()
