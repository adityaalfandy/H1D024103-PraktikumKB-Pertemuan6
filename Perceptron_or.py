# Import library
import numpy as np
import Perceptron as p

# Inisialisasi input dan target
X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
t = np.array([[1], [1], [1], [-1]])

# Pemanggilan model Perceptron
model = p.Perceptron(alpha=0.1, epoch=10)
model.fit(X, t)