from typing import Callable

import numpy as np
from numpy.random import Generator

Epoch = int

RowDiff = np.ndarray[int]
ColDiff = np.ndarray[int]
Distance = np.ndarray[float]

RecordArray = np.ndarray[float]
ResultMap = np.ndarray[list]

def kohonen(k: int,
            iterations: int,
            records: RecordArray,
            distance: Callable[[RowDiff,ColDiff], Distance],
            radius: Callable[[Epoch], float],
            eta: Callable[[Epoch], float],
            example: bool = False,
            rng: Generator = np.random.default_rng()
            ) -> ResultMap:
    activations_out = np.zeros(shape=(k,k), dtype=np.float64)
    col_matrix, row_matrix = np.meshgrid(np.arange(k), np.arange(k))

    result = np.empty(shape=(k,k), dtype=list)
    for i in range(k):
        for j in range(k):
            result[i,j] = list()

    if example:
        weights = rng.choice(records, size=(k,k))
    else:
        weights = rng.uniform(size=(k,k,records.shape[1]))

    
    for epoch, record in enumerate(rng.choice(records, size=iterations)):
        activations = np.dot(weights, record, out=activations_out)

        winner_index = np.argmax(activations)
        winner_col, winner_row = np.unravel_index(winner_index, activations.shape)

        result[winner_row, winner_col].append(record)
        
        distances = distance(row_matrix - winner_row, col_matrix - winner_col)

        for row, col in np.transpose(np.nonzero(distances < radius(epoch))):
            if row != winner_row or col != winner_col:
                weights[row, col] += eta(epoch) * (record - weights[row, col])
            
    return result
    