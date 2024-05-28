from typing import Callable, Optional

import numpy as np

from pandas import DataFrame
from numpy import ndarray
from numpy.random import Generator

Epoch = int

RowDiff = ndarray
ColDiff = ndarray
Distance = ndarray

RecordsMatrix = ndarray
WeightsMatrix = ndarray

def get_winner(weights: ndarray, record: ndarray) -> tuple[int,int]:
    activations = np.dot(weights, record)
    index = np.argmax(activations)
    return np.unravel_index(index, activations.shape)

def kohonen(k: int,
            iterations: int,
            df: DataFrame,
            distance: Callable[[RowDiff,ColDiff], Distance],
            radius: Callable[[Epoch], float],
            eta: Callable[[Epoch], float],
            example: bool = False,
            label: Optional[str] = None,
            rng: Generator = np.random.default_rng()
            ) -> tuple[RecordsMatrix, WeightsMatrix]:
    col_matrix, row_matrix = np.meshgrid(np.arange(k), np.arange(k))

    df_numeric = df if label is None else df.loc[:, df.columns != label]

    # Initialize weights
    if example:
        weights = rng.choice(df_numeric.values, size=(k,k))
    else:
        weights = rng.uniform(size=(k,k,df_numeric.shape[1]))

    # Train neurons
    for epoch, record_numeric in enumerate(rng.choice(df_numeric.values, size=iterations)):
        winner_row, winner_col = get_winner(weights, record_numeric)
        
        distances = distance(row_matrix - winner_row, col_matrix - winner_col)

        for row, col in np.transpose(np.nonzero(distances < radius(epoch))):
            weights[row, col] += eta(epoch) * (record_numeric - weights[row, col])
            
    # Initialize return matrix
    records_matrix = np.empty(shape=(k,k), dtype=list)
    for i in range(k):
        for j in range(k):
            records_matrix[i,j] = list()

    # Associate records
    for record, record_numeric in zip(df.values, df_numeric.values):
        winner_row, winner_col = get_winner(weights, record_numeric)

        records_matrix[winner_row, winner_col].append(record)
    
    return (records_matrix, weights)
    