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

def extract_average(col: int, result: ResultMap) -> np.ndarray[float]:
    return np.vectorize(lambda records: np.average([rec[col] for rec in records]))(result)

if __name__ == "__main__":
    df = pd.read_csv('./data/europe.csv')
    df_numeric = df.loc[:, df.columns != 'Country']
    df_normalized = (df_numeric - df_numeric.mean()) / df_numeric.std()

    values = df_normalized.values
    inputs = values.shape[1]

    print(df_normalized)
    
    result_map = kohonen(
        k=5,
        iterations=500 * inputs,
        records=values,
        distance=euclidean,
        radius=constant(2),
        eta=constant(0.7),
        example=True
    )

    heatmap = np.vectorize(len)(result_map)
    # sn.heatmap(heatmap)
    _, axs = plt.subplots(2, 4)

    axs[0,0].set_title('Area')
    axs[0,1].set_title('GDP')
    axs[0,2].set_title('Inflation')
    axs[0,3].set_title('Life.expect')
    axs[1,0].set_title('Military')
    axs[1,1].set_title('Pop.growth')
    axs[1,2].set_title('Unemployment')

    sn.heatmap(extract_average(0, result_map), ax=axs[0,0])
    sn.heatmap(extract_average(1, result_map), ax=axs[0,1])
    sn.heatmap(extract_average(2, result_map), ax=axs[0,2])
    sn.heatmap(extract_average(3, result_map), ax=axs[0,3])
    sn.heatmap(extract_average(4, result_map), ax=axs[1,0])
    sn.heatmap(extract_average(5, result_map), ax=axs[1,1])
    sn.heatmap(extract_average(6, result_map), ax=axs[1,2])

    plt.show()
