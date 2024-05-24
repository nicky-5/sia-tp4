import numpy as np

class hopfield:
    def __init__(self, pattern: np.ndarray, input: np.ndarray):
        if not self._is_stable(pattern):
            raise ValueError("Pattern is not stable")
        self.P = pattern.shape[0]                       # Number of patterns
        self.W = np.dot(pattern,pattern.T) / self.P     # Weight matrix
        np.fill_diagonal(self.W,0)                      # Diagonal to 0
        self.S = [input]                                # State of the network
        self.N = self.S[-1].shape[0]                    # Number of neurons
        self.t = 0                                      # Time step
        print(f"W shape: {self.W.shape}")
        print(f"S shape: {self.S[-1].shape}")

    @property
    def energy(self)->float:
        return -0.5 * np.einsum('ij,i,j', self.W, self.S[-1], self.S[-1])
    
    @property
    def conerged(self) -> bool:
        return (len(self.S) >= 2 and np.array_equal(self.S[-1], self.S[-2])) or self.t > 1000
    
    def train(self):
        while not self.conerged:
            self.S.append(self.activation_function)
            self.t += 1
        
        return self.S[-1], self.S, self.energy, self.t
    
    # checkeo si los vectores son ortogonales
    def _is_stable(self, vectors: np.ndarray):
        matrix = np.dot(vectors.T, vectors)
        np.fill_diagonal(matrix, 0)
        print(f"VECTOR: {vectors}")
        print(f"MATRIX: {matrix}")
        return np.allclose(matrix, np.zeros(matrix.shape),0)
    
    @property
    def activation_funtion(self):
        h = np.zeros(self.N)
        print(f" h shape: {h.shape}")

        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    h[i] += self.W[i,j] * self.S[-1][j]
        
        return np.sign(h)