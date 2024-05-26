import numpy as np

def letter_matrix(letters):
        matrix = []
        for letter in letters:
            with open(f"../data/{letter.lower()}.txt", "r") as f:
                parsed_letter = [[1 if char == '*' else -1 for char in line.strip("\n")] for line in f]
        
            matrix.append(parsed_letter)

        return matrix    