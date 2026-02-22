import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Aplica softmax linha a linha
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """
    Implementação do Scaled Dot-Product Attention
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    d_k = K.shape[1]

    # Produto escalar entre Q e K^T
    scores = np.dot(Q, K.T)

    # Scaling factor
    scaled_scores = scores / np.sqrt(d_k)

    # Softmax linha a linha
    attention_weights = softmax(scaled_scores)

    # Multiplicação pelos valores
    output = np.dot(attention_weights, V)

    return output, attention_weights