import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import random
import numpy as np

def letter_matrix(letters):
        matrix = []
        for letter in letters:
            with open(f"./data/letters/{letter.lower()}.txt", "r") as f:
                parsed_letter = [[1 if char == '*' else -1 for char in line.strip("\n")] for line in f]
        
            matrix.append(parsed_letter)

        return matrix

def rotate(letter):
    return np.array([[value for value in row] for row in letter]).T

def apply_noise(parsed_letter: list):
        """Adds noise to a given letter."""
        return [[-value if random.random() < 0.2 else value for value in row] for row in parsed_letter]

def hopfield_gif(initial_state, states_vector, width, height, duration):
	# Convertir la matriz inicial en una imagen
	first_image = Image.new('RGB', (len(initial_state[0]) * width, len(initial_state[0]) * height))
	for i in range(len(initial_state)):
		for j in range(len(initial_state[0])):
			color = 'black' if initial_state[i][j] == 1 else 'white'
			first_image.paste(color, (j * width, i * height, (j + 1) * width, (i + 1) * height))

	# Crear una lista de imágenes para el GIF
	gif_images = [first_image]

	#counter = 2
	# Reconstruir las matrices del vector y agregarlas al GIF
	for state in states_vector:
		matriz = np.reshape(state, (len(initial_state[0]), len(initial_state[0])))
		new_image = Image.new('RGB', (len(matriz[0]) * width, len(matriz) * height))
		for i in range(len(matriz)):
			for j in range(len(matriz[0])):
				color = 'black' if matriz[i][j] == 1 else 'white'
				new_image.paste(color, (j * width, i * height, (j + 1) * width, (i + 1) * height))
		gif_images.append(new_image)
		#new_image.save(f'{settings.Config.output_path}/hopfield{counter}.png')
		#counter+=1

	# Guardar las imágenes en un archivo GIF
	first_image.save(f'./output/hopfield.gif', save_all=True, append_images=gif_images[1:], optimize=False, duration=duration, loop=0)

def print_matrix(matrix):
    for row in matrix:
        row_str = ' '.join(['*' if val == 1 else ' ' for val in row])
        print(row_str)



def create_heatmap(matrix):
    if matrix.size != 25:
        raise ValueError("The input matrix must be 5x5.")
    
    # Reshape the flat matrix into a 5x5 array
    matrix = matrix.reshape(5, 5)
    
    # Create the heatmap
    plt.figure(figsize=(5, 5))
    sns.heatmap(matrix, annot=False, cbar=False, square=True, linewidth=2, linecolor='black', cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.show()