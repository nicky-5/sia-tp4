import h_utils as utils
import numpy as np
from src.hopfield import hopfield


if __name__ == "__main__":

    letters = ['O', 'Q', 'T', 'X']
    
    matrix = utils.letter_matrix(letters)

    print(matrix)

    for letter in matrix:
        print(f"Letter: {letter}")
        parsed_letter_rotated = utils.apply_noise(letter)

        flatted_patterns = np.column_stack([pattern.flatten() for pattern in np.array(utils.letter_matrix(letters))])
        flated_noise_pattern = np.array(parsed_letter_rotated).flatten().T

        hopfield_model = hopfield(flatted_patterns, flated_noise_pattern)

        S_f,S,energy,iterations = hopfield_model.train()

        utils.hopfield_gif(parsed_letter_rotated, S, 40, 40, 500)

        print(f"Energy: {energy}")
        print(f"Iterations: {iterations}")
        utils.create_heatmap(S_f)
    