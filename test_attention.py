import numpy as np
from attention import scaled_dot_product_attention

Q = np.array([[1, 0, 1]])
K = np.array([[1, 0, 1],
              [0, 1, 0]])
V = np.array([[1, 2],
              [3, 4]])

output, weights = scaled_dot_product_attention(Q, K, V)

print("Pesos de Atenção:")
print(weights)

print("\nSaída:")
print(output)